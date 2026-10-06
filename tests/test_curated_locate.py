import pytest

from src.curated.locate import HELP_EXCEL, HELP_FOLDER, find_annotations, find_violence_root, locate


def tree(tmp_path):
    (tmp_path / "violence" / "ASSASSINATION").mkdir(parents=True)
    (tmp_path / "VAW_results").mkdir()
    (tmp_path / "VAW_results" / "FYP_Annotations.csv").write_text("a,b\n", encoding="utf-8")
    return tmp_path


def test_finds_the_shortcut_folder_even_one_level_down(tmp_path):
    t = tree(tmp_path)
    assert find_violence_root(t) == t / "violence"
    (t / "violence").rename(t / "Violence")
    assert find_violence_root(t).name == "Violence"
    nested = tmp_path / "other" / "violence" / "STALKING"
    nested.mkdir(parents=True)
    assert find_violence_root(tmp_path / "other") == tmp_path / "other" / "violence"
    assert find_violence_root(tmp_path / "missing") is None


def test_annotation_file_search_prefers_the_excel(tmp_path):
    t = tree(tmp_path)
    assert find_annotations([t / "VAW_results"]).name == "FYP_Annotations.csv"
    (t / "FYP_data_filter.xlsx").write_bytes(b"x")
    assert find_annotations([t / "VAW_results", t]).name == "FYP_data_filter.xlsx"
    assert find_annotations([t / "nowhere"]) is None


def test_locate_gives_the_exact_instruction_when_something_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Add shortcut"):
        locate(tmp_path / "empty")
    t = tree(tmp_path)
    dr, ann, msgs = locate(t, results_dir=t / "VAW_results")
    assert dr == t / "violence" and ann.name == "FYP_Annotations.csv" and len(msgs) == 2
    (t / "VAW_results" / "FYP_Annotations.csv").unlink()
    with pytest.raises(FileNotFoundError, match="start-time file"):
        locate(t, results_dir=t / "VAW_results")
    assert "Shared with me" in HELP_FOLDER and "xlsx" in HELP_EXCEL
