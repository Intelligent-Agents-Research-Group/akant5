"""Tiny local tokenizer fixture for the retained workflow checks."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Twi sentences with the features that matter: ɛ/ɔ, digraphs, subject
#: prefixes, the object pronoun `no`, and the sentence from the publication
#: plan's worked example.
TWI_SENTENCES = [
    "Anadwo biara ɔfrɛ no.",
    "Ɔkɔɔ sukuu no mu.",
    "Me din de Kwaku.",
    "Ɛyɛ me dɛ paa.",
    "Yɛn nyinaa bɛkɔ fie.",
    "Wo ho te sɛn?",
    "Mepɛ sɛ mekɔ dwam.",
    "Ɔbaa no rekɔ adwuma.",
    "Wɔn mma no nyinaa wɔ sukuu.",
    "Nnipa no kɔɔ asɔre mu nnɛ.",
    "Onipa biara nim sɛ ɛyɛ nokware.",
    "Abofra no didi ntɛm.",
    "Ɔhene no tenaa ase wɔ ahenfie.",
    "Yɛbɛtumi akɔ hɔ ɔkyena.",
    "Kwasiada yɛkɔ asɔre.",
]

ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She calls him every night without fail.",
    "Grammar describes what a language permits and forbids.",
    "This sentence exists only to give the tokenizer some English.",
    "Knowledge graphs encode entities and the relations between them.",
]


@pytest.fixture(scope="session")
def twi_sentences() -> list[str]:
    return list(TWI_SENTENCES)


@pytest.fixture(scope="session")
def english_sentences() -> list[str]:
    return list(ENGLISH_SENTENCES)


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory, twi_sentences, english_sentences):
    """A real, tiny AkanT5-layout tokenizer built from the fixture sentences.

    Session-scoped because SentencePiece training, even at this size, is the
    slowest thing in the suite.
    """
    from akanlm.tokenizer.train_spm import TokenizerConfig, train

    out = tmp_path_factory.mktemp("tokenizer")
    config = TokenizerConfig(
        name="tiny",
        vocab_size=400,
        language_config=str(REPO_ROOT / "configs" / "languages" / "akan.json"),
    )
    corpus = (twi_sentences * 60) + (english_sentences * 20)
    train(config, out, sentence_iterator=corpus)

    from akanlm.tokenizer.train_spm import load

    return load(out)
