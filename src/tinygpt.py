
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from transformer_blocks import Block

MODE = "train"

# ==================================================
# PILIH TOKENIZER
# ==================================================

TOKENIZER_TYPE = "word"
# pilihan:
# "char"
# "word"
# "bpe"

# ==================================================
# LOAD CORPUS
# ==================================================

CORPUS_PATH = "../data/corpus.txt"

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read().lower()

# ==================================================
# TOKENIZATION
# ==================================================

if TOKENIZER_TYPE == "char":

    chars = sorted(list(set(text)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join([itos[i] for i in ids])

    data = torch.tensor(
        encode(text),
        dtype=torch.long
    )

    vocab_size = len(chars)

elif TOKENIZER_TYPE == "word":

    words = text.split()

    vocab = sorted(list(set(words)))

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}

    encode = lambda s: [stoi[w] for w in s.split()]
    decode = lambda ids: " ".join([itos[i] for i in ids])

    data = torch.tensor(
        [stoi[w] for w in words],
        dtype=torch.long
    )

    vocab_size = len(vocab)

elif TOKENIZER_TYPE == "bpe":

    spm.SentencePieceTrainer.Train(
        input="../data/corpus.txt",
        model_prefix="../tokenizer/tokenizer",
        vocab_size=300,
        model_type="bpe"
    )

    sp = spm.SentencePieceProcessor()
    sp.load("../tokenizer/tokenizer.model")

    encode = lambda s: sp.encode(
        s,
        out_type=int
    )

    decode = lambda ids: sp.decode(ids)

    data = torch.tensor(
        encode(text),
        dtype=torch.long
    )

    vocab_size = sp.get_piece_size()

else:
    raise ValueError("Tokenizer tidak dikenali")

print("Tokenizer :", TOKENIZER_TYPE)
print("Vocab Size:", vocab_size)
print("Dataset Size:", len(data))

# ==================================================
# HYPERPARAMETER
# ==================================================

block_size = 64
embedding_dim = 64
n_heads = 4
n_layers = 4

lr = 1e-3
epochs = 5000

# ==================================================
# BATCH
# ==================================================

def get_batch(batch_size=16):

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack(
        [data[i:i+block_size] for i in ix]
    )

    y = torch.stack(
        [data[i+1:i+block_size+1] for i in ix]
    )

    return x, y

# ==================================================
# MODEL
# ==================================================

class TinyGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    embedding_dim,
                    block_size,
                    n_heads
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(
            embedding_dim
        )

        self.head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(
            torch.arange(
                T,
                device=idx.device
            )
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T)
            )

        return logits, loss

    def generate(
        self,
        idx,
        max_new_tokens=150,
        temperature=0.8
    ):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(
                logits / temperature,
                dim=-1
            )

            next_idx = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_idx),
                dim=1
            )

        return idx

# ==================================================
# TRAIN
# ==================================================

if MODE == "train":

    model = TinyGPT()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    for step in range(epochs):

        xb, yb = get_batch()

        logits, loss = model(
            xb,
            yb
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 300 == 0:

            print(
                f"Step {step}, loss={loss.item():.4f}"
            )

    torch.save(
        model.state_dict(),
        f"../models/{TOKENIZER_TYPE}_model.pth"
    )

elif MODE == "generate":

    model = TinyGPT()

    model.load_state_dict(
        torch.load(
            f"../models/{TOKENIZER_TYPE}_model.pth",
            map_location="cpu"
        )
    )

    model.eval()


# ==================================================
# GENERATE
# ==================================================

prompt = "lari merupakan"

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long
)

out = model.generate(
    context,
    max_new_tokens=150,
    temperature=0.8
)

print("\nGenerated Text:\n")

print(
    decode(
        out[0].tolist()
    )
)

