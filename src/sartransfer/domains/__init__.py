"""New domains beyond glacier calving fronts (2026-09-26): glacial lakes, snow,
forests, alpine glaciers.

Same question as the glacier study -- do frozen encoders trained on ordinary or
satellite images transfer to SAR -- tested on public Sentinel-1 datasets, with
nothing tuned per domain.

    prepared.py    the compact on-disk format every converter writes (once, CPU)
    s1input.py     one fixed rule: two SAR channels -> encoder input + skip image
    run.py         frozen features -> linear probe + small decoder -> metrics
    embed_views.py what the frozen features look like (notebook 03), trains nothing
    hub.py         private Hugging Face copies of the converted data; staging to local disk
    glacial_lakes.py, snow.py, forest.py, alpine.py
                   one converter per public dataset: download, inventory, convert
"""
