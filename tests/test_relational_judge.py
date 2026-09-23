import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from relational_judge import judge

EDGES = {
    ("Nex-41", "maker"): "Lina",
    ("Bolt-7", "maker"): "Omar",
    ("Kite-3", "maker"): "Vera",
    ("Orb-12", "maker"): "Hugo",
    ("Nex-41", "city"): "Riga",
    ("Bolt-7", "city"): "Oslo",
    ("Kite-3", "city"): "Bern",
    ("Orb-12", "city"): "Lyon",
    ("Nex-41", "color"): "amber",
    ("Bolt-7", "color"): "cobalt",
    ("Kite-3", "color"): "ivory",
    ("Orb-12", "color"): "scarlet",
    ("Nex-41", "tool"): "chisel",
    ("Bolt-7", "tool"): "spindle",
    ("Kite-3", "tool"): "kiln",
    ("Orb-12", "tool"): "loom",
}


def row(kind, answer, **query):
    query["kind"] = kind
    return {"answer": answer, "query": query}


def test_direct_and_triple_pass():
    item = row("atomic", "Lina", subject="Nex-41", relation="maker")
    assert judge("Lina", item, EDGES)["verdict"] == "PASS"
    assert judge("Nex-41 maker is Lina.", item, EDGES)["verdict"] == "PASS"


def test_wrong_subject_fails():
    item = row("atomic", "Lina", subject="Nex-41", relation="maker")
    assert judge("Bolt-7 maker is Lina.", item, EDGES)["verdict"] == "FAIL"


def test_one_wrong_of_two_fails():
    item = row(
        "composition",
        "Lina and Bern",
        subject_a="Nex-41",
        relation_a="maker",
        subject_b="Kite-3",
        relation_b="city",
    )
    assert judge("Lina and Oslo", item, EDGES)["verdict"] == "FAIL"
    assert judge("Nex-41 maker is Lina and Kite-3 city is Oslo.", item, EDGES)["verdict"] == "FAIL"


def test_both_triples_pass_without_exact_string():
    item = row(
        "composition",
        "Lina and Bern",
        subject_a="Nex-41",
        relation_a="maker",
        subject_b="Kite-3",
        relation_b="city",
    )
    verdict = judge("Nex-41 maker is Lina and Kite-3 city is Bern.", item, EDGES)
    assert verdict["verdict"] == "PASS"


def test_truncated_abstains():
    item = row(
        "composition",
        "Lina and Bern",
        subject_a="Nex-41",
        relation_a="maker",
        subject_b="Kite-3",
        relation_b="city",
    )
    verdict = judge("Nex-41 maker is Lina and Kite-3", item, EDGES, hit_limit=True)
    assert verdict["verdict"] == "ABSTAIN"
    assert verdict["reason"] == "truncated"


def test_repetition_is_only_the_value():
    item = row("atomic", "Lina", subject="Nex-41", relation="maker")
    assert judge("Lina Lina Lina", item, EDGES)["verdict"] == "PASS"
    item = row(
        "composition",
        "Lina and Bern",
        subject_a="Nex-41",
        relation_a="maker",
        subject_b="Kite-3",
        relation_b="city",
    )
    assert judge("Lina Lina Lina", item, EDGES)["verdict"] == "FAIL"


def test_negation_then_contradiction_fails():
    item = row("false_claim", "false", subject="Nex-41", relation="maker", claimed="Omar")
    assert judge("No, the maker of Nex-41 is Omar.", item, EDGES)["verdict"] == "FAIL"
    assert judge("No, the maker of Nex-41 is Lina.", item, EDGES)["verdict"] == "PASS"
    assert judge("False. Nex-41 maker is Lina.", item, EDGES)["verdict"] == "PASS"


def test_claimed_value_alone_is_not_taken_from_the_question():
    item = row("false_claim", "false", subject="Nex-41", relation="maker", claimed="Omar")
    assert judge("Omar", item, EDGES)["verdict"] == "FAIL"


def test_digit_junk_after_false_is_not_a_negation():
    item = row("false_claim", "false", subject="Nex-41", relation="maker", claimed="Omar")
    verdict = judge("false-6-1-1-1", item, EDGES)
    assert verdict["verdict"] == "ABSTAIN"


def test_extra_text_after_a_direct_answer_abstains_when_not_a_triple():
    item = row("atomic", "Lina", subject="Nex-41", relation="maker")
    verdict = judge("Lina The sky is blue", item, EDGES)
    assert verdict["verdict"] == "ABSTAIN"
