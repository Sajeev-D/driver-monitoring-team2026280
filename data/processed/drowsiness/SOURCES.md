# Sources and sharing terms

These tables contain derived research features. They do not replace the source
datasets' terms, establish redistribution permission, or provide a new licence
for the underlying measurements. Check the team's permission before publishing
or redistributing the CSVs. Raw participant images/videos are excluded.

## DROZY

- [Official dataset/publication](https://orbi.uliege.be/handle/2268/191620)
- [DROZY licence, 18 July 2018](https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf)
- The supplied licence limits use to research/evaluation, excludes commercial
  development, and requires prior written approval to redistribute the database
  in full or part. It also restricts showing participant images. Packaging
  derived features here is not a determination that sharing them is permitted.
- Required citation: Quentin Massoz, Thomas Langohr, Clémentine François and
  Jacques G. Verly, “The ULg Multimodality Drowsiness Database (called DROZY)
  and Examples of Use,” WACV 2016. DOI: 10.1109/WACV.2016.7477715.

## UTA-RLDD

- [Official dataset description and access](https://sites.google.com/view/utarldd/home)
- Citation: Reza Ghoddoosian, Marnim Galib and Vassilis Athitsos,
  “A Realistic Dataset and Baseline Temporal Model for Early Drowsiness
  Detection,” CVPR Workshops 2019. [Paper](https://arxiv.org/abs/1904.07312).
- Follow the source dataset's access, publication and attribution conditions.
  Only a subset of participants consented to publication of face images; no
  images are included here. A Kaggle download does not by itself establish a
  new licence for the underlying dataset or its derived tables.

## Feature provenance

The common schema and reproducible exporters are in `src/eye_features/` and
`scripts/`. Counts, missingness and file checksums are in `manifest.json`.
Exact subject/session identifiers are preserved for grouping and source
traceability; actual identities are not provided. Original KSS/class labels
are retained with explicit label rules. No sleep history, physiological signal
or precise KSS measurement was invented where a source did not supply it.
