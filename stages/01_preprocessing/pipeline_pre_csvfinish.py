import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import time
    return pl, time


@app.cell
def _():
    dir = "/mnt/Disk2/zeitenwende/huang/output/gesamt/punkt/"
    return (dir,)


@app.cell
def _(dir, pl):
    pl.scan_csv(f'{dir}kwic_hits_zeitenwende_punkt.csv').select(pl.len()).collect().item()
    return


@app.cell
def _(dir, pl):
    pl.scan_csv(f'{dir}kwic_hits_zeitenwende_punkt.csv').head(10).collect()
    return


@app.cell
def _(dir, pl, time):
    start = time.time()

    (
        # infer_schema=False: all columns as text, no type parsing (also keeps IDs like "102605236x" intact)
        pl.scan_csv(f'{dir}kwic_hits_zeitenwende_punkt.csv', infer_schema=False)
            .with_columns(
                #(
                #    (pl.col("context_before").fill_null("") + " " + pl.col("context_after").fill_null(""))
                #).alias("context"),
                # year and quarter straight from the ISO date string "YYYY-MM-DD", without date parsing
                pl.format(
                    "{}-{}",
                    pl.col("date").str.slice(0, 4),
                    (pl.col("date").str.slice(5, 2).cast(pl.Int8) + 2) // 3
                ).alias("slice")
            )
        # columns not selected (position, sentence_based, date) are dropped by select
        #.select("id", "target_form", "context", "slice", "facet_1", "context_before", "context_after")
        .select("id", "target_form", "context_before", "context_after", "slice", "facet_1")
        .sink_csv(f'{dir}export_ns_ba.csv')
    )

    print(time.time() - start, "Sekunden")
    return


@app.cell
def _(dir, pl):
    pl.scan_csv(f'{dir}zeitenwende_wh_export_ns_ba.csv').head(10).collect()
    return


@app.cell
def _(dir, pl):
    lf = pl.scan_csv(f'{dir}zeitenwende_wh_export_ns_ba.csv', schema_overrides={"facet_1":pl.String})
    return (lf,)


@app.cell
def _(lf, pl):
    result = (
        lf
        .filter(pl.col("facet_1") == "1025922700")
        .head(500)
        .collect(engine="streaming")
    )
    result
    return


if __name__ == "__main__":
    app.run()
