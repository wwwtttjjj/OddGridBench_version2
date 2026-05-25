# Dataset Image Statistics

Count definitions:
- `initial_image_files`: raw/source images found in the dataset split, preferably under `A_cropped_images`.
- `iol_samples`: number of IOL metadata records; one record normally corresponds to one grid image.
- `iol_generated_image_files`: actual generated IOL image files on disk.
- `iol_source_image_slots`: total source-cell slots used by IOL samples.
- `soi_samples`: number of SOI metadata records; one record can contain multiple images/icons.
- `soi_generated_image_files`: actual SOI image files on disk, counted recursively.
- `soi_total_icons`: sum of `total_icons`/source image details in SOI metadata.

| dataset | split | initial_image_files | iol_samples | iol_generated_image_files | iol_source_image_slots | soi_samples | soi_generated_image_files | soi_total_icons | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTech_Dataset_transformed | train | 598 | 122 | 122 | 2030 | 130 | 1621 | 1621 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| GOODADS | train | 1520 | 411 | 411 | 7000 | 431 | 5556 | 5556 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| MPDD | train | 151 | 40 | 40 | 641 | 41 | 508 | 508 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| MVTEC_LOCO | train | 761 | 143 | 143 | 2388 | 152 | 1929 | 1929 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| RAD | train | 3134 | 1342 | 1342 | 21583 | 1371 | 17048 | 17048 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| VisA | train | 3400 | 225 | 225 | 3709 | 253 | 3145 | 3145 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| mvtec | train | 3277 | 502 | 502 | 8204 | 490 | 6135 | 6135 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| mvtec_ad2 | train | 472 | 68 | 68 | 1184 | 62 | 782 | 782 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| BTech_Dataset_transformed | test | 890 | 53 | 53 | 891 | 65 | 890 | 890 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| GOODADS | test | 3240 | 310 | 310 | 4036 | 431 | 3240 | 3240 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| MPDD | test | 538 | 43 | 43 | 541 | 52 | 538 | 538 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| RAD | test | 81 | 7 | 7 | 84 | 8 | 81 | 81 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| SimplifiedChinese | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | initial=not_found |
| VisA | test | 3406 | 193 | 193 | 3414 | 247 | 3406 | 3406 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| hanzi | test | 244 | 200 | 200 | 7362 | 200 | 2775 | 2775 | initial=hanzi_png; iol=iol_test_data/iol_test_data.json; soi=soi_test_data/soi_test_data.json |
| mnist | test | 60000 | 200 | 200 | 7229 | 200 | 2761 | 2761 | initial=mnist_png; iol=iol_test_data/iol_test_data.json; soi=soi_test_data/soi_test_data.json |
| mvtec | test | 2011 | 126 | 126 | 2043 | 146 | 2011 | 2011 | initial=A_cropped_images; iol=A_iol_type_data/all_iol_combined_metadata.json; soi=A_soi_type_data/all_soi_combined_metadata.json |
| mvtec_loco | test | 0 | 195 | 195 | 2925 | 0 | 0 | 0 | initial=not_found; iol=iol_test_data/iol_test_data.json |
| Synthetic_IOL | train |  | 10000 | 10000 |  |  |  |  | create_data; no explicit initial source-image pool |
| Synthetic_SOI | train |  |  |  |  | 10000 | 159986 | 159986 | create_data; no explicit initial source-image pool |
| Synthetic_IOL | val |  | 400 | 400 |  |  |  |  | create_data; no explicit initial source-image pool |
| Synthetic_SOI | val |  |  |  |  | 400 | 6500 | 6500 | create_data; no explicit initial source-image pool |
| Synthetic_IOL | test |  | 400 | 400 |  |  |  |  | create_data; no explicit initial source-image pool |
| Synthetic_SOI | test |  |  |  |  | 400 | 6428 | 6428 | create_data; no explicit initial source-image pool |
| TOTAL | test | 70410 | 1727 | 1727 | 28525 | 1749 | 22130 | 22130 | sum of numeric columns |
| TOTAL | train | 13313 | 12853 | 12853 | 46739 | 12930 | 196710 | 196710 | sum of numeric columns |
| TOTAL | val | 0 | 400 | 400 | 0 | 400 | 6500 | 6500 | sum of numeric columns |
| TOTAL | all | 83723 | 14980 | 14980 | 75264 | 15079 | 225340 | 225340 | sum of numeric columns |
