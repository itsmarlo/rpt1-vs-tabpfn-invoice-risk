from src.evaluate import save_skipped_plot


def test_save_skipped_plot_writes_png(tmp_path) -> None:
    output_path = tmp_path / "skipped.png"

    save_skipped_plot(output_path, "model skipped")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
