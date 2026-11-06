# Upstream source notices

This directory preserves exact license and notice files from the pinned requests,
Flask, pylint and pytest snapshots used by the benchmark. Saved model/tool records
and generated reports may contain excerpts inspected from those repositories.
The upstream notices accompany those excerpts separately from the harness and
adapter licenses.

[provenance.json](provenance.json) maps each retained file to its original filename,
repository path, base commit, public source URL and benchmark task. Identical bytes
are stored once under `files/<sha256>/<original-filename>`; every source association
remains in the provenance record. The files themselves have not been rewritten.

The prepared snapshot records and committed evaluation manifest agree on the base
commits. The preparation code checks tracked source content against each base
commit before removing Git history. Notice collection did not repeat that Docker
verification; the provenance states this distinction explicitly.

The cached, pinned SWE-bench Verified dataset card is retained as source metadata.
It and the locally cached snapshot do not contain a license declaration or license
file. This observation does not establish permission to reuse the dataset, and
the adapter's license is not offered as a substitute for missing dataset terms.
No dataset rows, reference fixes or evaluation test patches are copied here.

The report command copies this notice bundle into its report and dashboard output.
The checkpoint export copies it into the Hugging Face publication bundle and
includes the copied files in its SHA256 inventory.
