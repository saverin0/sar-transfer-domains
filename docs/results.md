# Results in full

This page holds the method details, the full results table (validation included), the
frozen-feature numbers of notebook 03, the published results in detail and every caveat.
The short version is in the [README](../README.md).

## Protocol (fixed before any number)

The protocol was written down on 2026-09-26, before any result on these datasets existed.
Nothing is tuned per dataset.

- Encoders and layers come from part one's glacier study, DINOv3 ViT-L SAT-493M at layer 21 and
  C-RADIOv4-H at layer 32 (chosen there on the glacier validation split, not here).
- Input mode `vv_vh_diff` is co-pol, cross-pol and co minus cross, standardised with the training
  split's channel statistics. Missing values become 0 after standardisation. The decoder's image
  skip is clip((z_co + 3) / 6, 0, 1) as 8 bit, which does not depend on the units.
- One no-data rule for every dataset. A pixel whose co-pol or cross-pol value is missing is
  ignored (label 255) in training and scoring.
- Head 1, the linear probe. One linear layer on the 16 px patch features. The patch label is the
  majority vote of its valid pixels. Balanced cross-entropy, AdamW with learning rate 1e-3,
  batch 65,536 patches, at most 2,000,000 training patches (uniform sample, seed 0), at least
  30 epochs and at least about 1,000 optimiser steps (the glacier probe's step count, so small
  datasets are not trained for only a few dozen steps). Pixel predictions are the bilinearly
  upsampled patch logits.
- Head 2, the small decoder. Part one's row 3 (grid decoder with the image skip), 4,000 steps,
  batch 8, AdamW with learning rate 1e-3 and a one-cycle schedule, balanced cross-entropy that
  ignores 255, horizontal flips, seeds 0, 1 and 2, at most 4,000 training chips (uniform sample,
  seed 0).
- Official splits. Only "train" is trained on. Nothing is selected on any split, so validation
  and every test split are scored in the same run; validation is reported as a check only.
- Metrics per split come from one pixel confusion matrix, per-class IoU, mIoU and accuracy; for
  the two-class datasets also IoU, F1, precision and recall of the target class; per region
  (pooled); a majority baseline that predicts the most frequent training class. Every result
  row carries the SAR units of its dataset.

## Datasets and conversion

Notebook 01 (CPU runtime with High-RAM) downloads each dataset from its source, checks the
published checksums (alpine glaciers by byte counts, see below), converts it once into one compact format (radar in float16 with missing
values as NaN, labels as uint8 with 255 = ignore, per-chip metadata with region and date) and
copies it to Drive.

| dataset | chips per split | chip size | units |
|---|---|---|---|
| glacial lakes | 14,820 train, 1,851 val, 1,834 test, 613 test_region_SEE, 1,105 test_challenge | 256 px | per-chip 0-1 stretch |
| snow | 3,618 train, 1,697 val, 1,373 test, 3,471 test_spatial, 6,766 test_temporal | 512 px | dB of uncalibrated intensity |
| forest (ForTy v1) | 5,052 train, 618 val, 636 test | 128 px | dB |
| alpine glaciers (GlaViTU, Alps) | 177 train, 59 val, 60 test | 992 px | dB |

- **Glacial lakes.** The main set comes from the Hugging Face copy of Glacial-Lake-Bench
  (Sk-21/Cryo-Bench) and is checked file by file against the Zenodo release; the challenge set
  comes from Zenodo. The official split is random, and 93 % of its test chips share a
  Sentinel-2 scene with training. So all chips of region SEE (South Asia East, central and
  eastern Himalaya) are held out as `test_region_SEE` and never trained on; training copies
  that are byte-identical to held-out chips are dropped. Bands 10 and 11 are VV and VH (HH and
  HV in Arctic regions, which cannot be told apart per chip).
- **Snow.** The 43 GB archive is read in place. Labels are MODIS snow cover without
  interpolation, in radar geometry.
- **Forest.** The first 32 train, 32 validation and 32 test shards of ForTy v1, read with an own
  TFRecord reader (no TensorFlow). Descending orbit, which has 84 % valid pixels against 69 %
  for ascending. Counted from each chip's latitude and longitude after conversion, Europe holds
  10.8 % of the chips (547 train, 68 val, 69 test), Germany and its neighbours 0.8-1.8 %. A
  Europe-only run was not done.
- **Alpine glaciers.** The Alps tiles of the GlaViTU dataset, copied over HTTP ranges (18.9 GB) and
  checked by byte counts (the published md5 covers whole files only), official random split. Backscatter is converted back from the stored min-max log10 values to
  dB; per tile the orbit with more valid pixels is used; one 992 px chip per tile (16.7 % is
  padding and ignored). Pixels on the (0, 0) outline code are ignored (the official evaluation
  counts them as non-glacier).

Two converter fixes were found by the real data, both covered by tests before the final
conversion.

- The Hugging Face copy of Glacial-Lake-Bench is the Zenodo release plus 6 extra test image and
  mask pairs (12 files). They are now listed in the identity record and never read, so the
  splits are exactly Zenodo's.
- The snow label files are float32 with 2 bands (3 for one label variant), not the 1-band uint8
  of the dataset's README. Band 1 holds the documented codes -1, 0 and 1 and is the label.
  Band 2 is undocumented; where it is not 1 the pixel becomes 255 and is counted. It was 1 in
  every pixel of every split. The radar values are all positive (about 6-56), i.e. dB of
  uncalibrated intensity, which the per-channel standardisation handles.

The ForTy class codes are inferred (no public source gives the integers). The order used is
0 unknown, 1 natural forest, 2 planted forest, 3 tree crops, 4 other vegetation, 5 built-up,
6 water, 7 ice, 8 bare, and the class shares are checked at run time.

**Alpine check before the run.** Only test chip 42 (tile ALP-31-33, descending) has values above
20 dB, 4 co-pol and 6 cross-pol pixels of 867,180 (up to 169.8 and 112.1 dB). The training
range ends at 14.5 dB. They are kept as converted.

## Run

Notebook 02 ran on a Colab A100 (80 GB) with High-RAM (179 GB system RAM). Two speed-ups were
added before the run and checked on toy datasets, where every result file was identical to the
older code. The pixel confusion is counted per chip on the GPU where the predictions are made
(the CPU count took about 16 min per encoder for snow alone), and each encoder is loaded once
for all datasets.

Time per encoder was forest 3-4 min, glacial lakes 5, snow 15-16 and alpine glaciers 23 (the
decoder takes 7.3 min per seed on the 992 px chips). All four datasets with both encoders took
about 1 h 35 min.

## Full results

For the two-class datasets the probe and decoder columns are the IoU of the target class (lake,
glacier, snow); for forest they are the mIoU over its 8 classes. The majority column is the
baseline's mIoU (its target-class IoU is 0). Decoder values are the mean ± SD of 3 seeds.

| dataset | split | encoder | majority mIoU | probe | decoder |
|---|---|---|---|---|---|
| glacial lakes | val | satellite DINOv3 | 0.478 | 0.278 | 0.288 ± 0.006 |
| glacial lakes | val | C-RADIO | 0.478 | 0.276 | 0.273 ± 0.002 |
| glacial lakes | test | satellite DINOv3 | 0.480 | 0.254 | 0.279 ± 0.004 |
| glacial lakes | test | C-RADIO | 0.480 | 0.259 | 0.260 ± 0.001 |
| glacial lakes | test_challenge | satellite DINOv3 | 0.480 | 0.304 | 0.299 ± 0.004 |
| glacial lakes | test_challenge | C-RADIO | 0.480 | 0.301 | 0.285 ± 0.002 |
| glacial lakes | test_region_SEE | satellite DINOv3 | 0.493 | 0.220 | 0.208 ± 0.002 |
| glacial lakes | test_region_SEE | C-RADIO | 0.493 | 0.196 | 0.179 ± 0.006 |
| alpine glaciers | val | satellite DINOv3 | 0.471 | 0.361 | 0.542 ± 0.009 |
| alpine glaciers | val | C-RADIO | 0.471 | 0.344 | 0.501 ± 0.009 |
| alpine glaciers | test | satellite DINOv3 | 0.471 | 0.355 | 0.543 ± 0.004 |
| alpine glaciers | test | C-RADIO | 0.471 | 0.355 | 0.507 ± 0.006 |
| snow | val | satellite DINOv3 | 0.237 | 0.625 | 0.689 ± 0.007 |
| snow | val | C-RADIO | 0.237 | 0.616 | 0.588 ± 0.001 |
| snow | test (Guil, 2018-19) | satellite DINOv3 | 0.182 | 0.640 | 0.735 ± 0.007 |
| snow | test (Guil, 2018-19) | C-RADIO | 0.182 | 0.622 | 0.602 ± 0.004 |
| snow | test_spatial (Gyronde) | satellite DINOv3 | 0.192 | 0.618 | 0.678 ± 0.010 |
| snow | test_spatial (Gyronde) | C-RADIO | 0.192 | 0.601 | 0.510 ± 0.017 |
| snow | test_temporal (Guil, 2019-20) | satellite DINOv3 | 0.214 | 0.673 | 0.770 ± 0.011 |
| snow | test_temporal (Guil, 2019-20) | C-RADIO | 0.214 | 0.644 | 0.591 ± 0.014 |
| forest | val | satellite DINOv3 | 0.039 | 0.402 | 0.377 ± 0.013 |
| forest | val | C-RADIO | 0.039 | 0.374 | 0.344 ± 0.010 |
| forest | test | satellite DINOv3 | 0.040 | 0.402 | 0.384 ± 0.009 |
| forest | test | C-RADIO | 0.040 | 0.391 | 0.350 ± 0.008 |

- With the decoder, the satellite DINOv3 is ahead of C-RADIO on every dataset and split. In
  part one's glacier test it was the other way round.
- The two linear probes are within 0.03 of each other everywhere.
- The decoder helps on alpine glaciers (both encoders) and on snow (satellite DINOv3), changes
  glacial lakes by -0.02 to +0.03 and loses 0.02-0.04 on forest, whose 128 px chips give the
  encoders only an 8 x 8 grid. C-RADIO's decoder stays below its own probe on snow.
- Validation is within 0.025 of test, except the satellite DINOv3 decoder on snow (0.689
  against 0.735).
- The held-out region SEE is the hardest glacial-lake split for both encoders (0.18-0.22),
  below the official test split, whose chips mostly share scenes with training.

## Frozen features (notebook 03)

Descriptive only, it trains nothing and runs after notebook 02 (it reuses 02's channel
statistics). For each dataset and encoder, on the test split, with the run's own input.

- **Embedding maps.** The patch features of two chips as colours (the first three principal
  components as RGB) and as k-means clusters.
- **Similarity search by example.** From one query patch, the most similar patches of the whole
  test split, and the share of the query's class among them.
- **Class distances.** For each class, how far its patches sit from the other classes, as a
  ranking AUC over the test patches (1 = fully apart), and histograms of each patch's margin
  (cosine similarity to its own class centre minus that to the nearest other class centre).

The chips and the query are picked by fixed rules from the labels only. A patch counts as pure
when at least 90 % of its labelled pixels are one class and at least half of it is labelled. A
chip is eligible when at least 90 % of its radar-covered area is labelled and that area is at
least half the chip. The number of similar patches is 100, capped at the number of pure patches
of the query's class minus 1.

The same views are repeated after AnyUp upsamples the features to 1/4 of the image resolution
(4 px blocks). AnyUp is guided by the decoder's image skip. The query is the mean unit vector of
the 16 blocks inside the query patch, the number of similar blocks is 16 times the raw number,
the class centres come from every pure block, and the AUCs use a fixed uniform sample of at most
400,000 pure blocks (seed 0).

| mean class AUC | satellite DINOv3 | satellite DINOv3, AnyUp | C-RADIO | C-RADIO, AnyUp |
|---|---|---|---|---|
| forest | 0.888 | 0.870 | 0.867 | 0.849 |
| glacial lakes | 0.911 | 0.886 | 0.909 | 0.890 |
| alpine glaciers | 0.795 | 0.754 | 0.757 | 0.711 |
| snow | 0.691 | 0.738 | 0.624 | 0.628 |

| similarity search, share of the query's class | satellite DINOv3 | C-RADIO |
|---|---|---|
| glacial lakes, lake query, 100 most similar patches | 0.83 | 0.37 |
| glacial lakes, lake query, 1,600 most similar AnyUp blocks | 0.97 | 0.44 |
| forest, built-up query, 100 most similar patches | 0.91 | 0.51 |
| forest, built-up query, 1,600 most similar AnyUp blocks | 0.78 | 0.60 |

- With the raw features, the satellite DINOv3 keeps the classes further apart on every dataset,
  which matches the decoder results. On glacial lakes it is nearly a tie (0.911 against 0.909),
  and after AnyUp C-RADIO is slightly ahead there (0.890 against 0.886).
- AnyUp lowers the mean AUC by 0.02-0.05 on forest, glacial lakes and alpine glaciers for both
  encoders and raises it on snow for the satellite DINOv3 (+0.05). The pure 4 px blocks include
  many cells next to class borders that no pure 16 px patch covers, so the AnyUp AUCs are over a
  different, harder set of cells, not a clean loss.
- On the alpine chips, C-RADIO's AnyUp k-means splits the chip into a left and a right half,
  while the satellite DINOv3's clusters follow the terrain.
- The class centres come from the test split's own labels, so these numbers describe the
  features; they are not scores of a model.

Run time on an A100 (80 GB) was 7.1 min for the raw views and 5.2 min for the AnyUp views.

## Published results in detail

Every published number below comes from a model trained for the task. All but the snow row also
use optical data. Ours are frozen encoders on the two radar channels alone.

- **Glacial lakes.** Kaushik et al., Glacial-Lake-Bench (ESSD preprint, in review), Tables 3 and 4.
  Test mIoU including background, Prithvi-EO-2.0 0.85, U-Net 0.82, DOFA 0.82, DeepLabv3+ 0.80;
  challenge set Prithvi-EO-2.0 and DOFA 0.79, U-Net 0.76. Their input has 11 channels (optical,
  SAR and elevation; Prithvi 6), the models are fully fine-tuned, one seed. Our best mIoU is
  0.595 on test (satellite DINOv3 decoder) and 0.614 on the challenge set (satellite DINOv3
  probe). Cryo-Bench's code repository (not its paper) lists frozen encoders on the same split
  with the 11 channels, mIoU 0.69-0.80; no radar-only run there.
- **Alpine glaciers.** Maslov et al. (Nature Communications 2025), supplementary tables. Alps
  test glacier IoU 0.844 for GlaViTU and 0.835 for DeepLabv3+ from optical data and elevation;
  0.873 for GlaViTU with Sentinel-1 backscatter and coherence added. Same split and metric as
  ours; our best is 0.543 (satellite DINOv3 decoder).
- **Forest.** Jiang and Neumann, ForTy (IGARSS 2025), Table I. Macro F1 81.1 for MTSViT, 79.6
  TSViT, 49.4 UTAE, 32.4 UNet3D, from seasonal Sentinel-2, climate and elevation on the full
  test set. Not comparable to our 8-class mIoU of 0.402 on 636 test chips.
- **Snow.** Briand et al. (ISPRS Journal of Photogrammetry and Remote Sensing 239, 2026), read
  in the open HAL copy (hal-05186268), Tables 3 and 6. A U-Net with EfficientNet-B0, 6 seeds,
  threshold tuned on validation for overall F1, overlapping 512 px predictions with Gaussian
  weighting, pooled counts over all test dates in radar geometry, snow as the positive class.
  On the Guil 2018-19 test with non-interpolated labels, channel set A (VV and VH only) reaches
  accuracy 0.872, overall F1 0.873, snow F1 0.897 (SD 0.004) and no-snow F1 0.831. Set B adds a
  snow-free reference image and reaches snow F1 0.921. With set B and interpolated labels the
  test snow F1 rises to 0.937-0.943. Only set A is radar-only, so only the Guil 2018-19 test
  compares like for like with ours.
- **Our snow F1.** F1 = 2 IoU / (1 + IoU) holds exactly when both come from the same pooled
  counts. The satellite DINOv3 decoder's snow IoU of 0.735 on test is a snow F1 of 0.847
  (0.808 on the other basin, 0.870 on the next winter, which have no radar-only published
  counterpart).

## Caveats

- Only the snow row compares like for like with a published result. The others use optical data,
  and forest uses another metric on a far larger test set.
- The glacial-lake values are the dataset's per-chip 0-1 stretch, not dB, so the third input
  channel is not a log ratio there. The official glacial-lake test split shares Sentinel-2
  scenes with training; the SEE hold-out is the true new-region test.
- Snow labels are MODIS snow cover in 500 m blocks, and 28-37 % of the chip pixels have no label
  (cloud gaps and missing radar).
- The ForTy class codes are inferred. The forest run uses 96 of the dataset's shards, not the
  full 1.1 TB.
- Ten pixels of one alpine test chip are far outside the training range (see the alpine check
  above) and are kept as converted.
- Decoder numbers are over 3 seeds, probe numbers over one fit. Published numbers use their own
  seeds and thresholds.
