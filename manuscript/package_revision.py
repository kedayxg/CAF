"""Compile and package a complete local revision; performs no submission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path(__file__).resolve().parent/"build")
    args = parser.parse_args()
    paper = Path(__file__).resolve().parent
    evidence = json.loads((paper.parent/"docs/revision-results.json").read_text())
    if evidence["status"] != "complete" or evidence["missing_tasks"]:
        raise SystemExit("Refusing to package an incomplete revision; finish reruns and build_revision.py first")
    validation_path = paper.parent/"docs/revision-validation.json"
    validation = json.loads(validation_path.read_text())
    if (validation.get('status') != 'passed'
            or validation.get('validated_source_files') != evidence['source_files']
            or validation.get('frozen_sha256') != evidence['frozen_sha256']
            or validation.get('main_experiment_sha256') != evidence['main_experiment_sha256']
            or validation.get('natural_reference') != evidence['natural_reference']
            or validation.get('gurobi_audit_sha256') != evidence.get('gurobi_audit_sha256')
            or not evidence.get('sampling_control_complete')
            or validation.get('sampling_control_sha256') != evidence.get('sampling_control_sha256')):
        raise SystemExit('Independent validation does not match these exact records and frozen snapshot')
    if evidence.get('primary_backend') == 'Gurobi':
        primary_path = paper.parent/'docs/gurobi-main-results.json'
        primary_validation_path = paper.parent/'docs/gurobi-main-validation.json'
        primary = json.loads(primary_path.read_text())
        primary_validation = json.loads(primary_validation_path.read_text())
        if (primary_validation['status'] != 'passed'
                or primary['source_files'] != primary_validation['validated_source_files']
                or hashlib.sha256(primary_path.read_bytes()).hexdigest() != evidence['gurobi_main_sha256']
                or hashlib.sha256(primary_validation_path.read_bytes()).hexdigest() != evidence['gurobi_main_validation_sha256']):
            raise SystemExit('Gurobi primary evidence does not match its independent validation')
    study = evidence.get('coordination_study')
    if not study:
        raise SystemExit('Missing BCD/capacity/subset follow-up evidence')
    study_path=paper.parent/'docs/coordination-study-results.json'
    study_validation_path=paper.parent/'docs/coordination-study-validation.json'
    study_result=json.loads(study_path.read_text())
    study_validation=json.loads(study_validation_path.read_text())
    if (study_validation['status']!='passed'
            or study_result['source_files']!=study_validation['validated_source_files']
            or hashlib.sha256(study_path.read_bytes()).hexdigest()!=study['results_sha256']
            or hashlib.sha256(study_validation_path.read_bytes()).hexdigest()!=study['validation_sha256']
            or hashlib.sha256(json.dumps(study_result['cases'],sort_keys=True).encode()).hexdigest()!=study_validation['summary_sha256']):
        raise SystemExit('Follow-up evidence does not match independent validation')
    args.build_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error",
        f"-outdir={args.build_dir.resolve()}","main.tex"],cwd=paper,check=True)
    log = (args.build_dir/"main.log").read_text()
    if "undefined references" in log or "undefined citations" in log or "Overfull" in log:
        raise SystemExit("Resolve undefined references/citations or overfull boxes before packaging")
    pdf = paper/"CAF_arxiv_revision.pdf"
    shutil.copyfile(args.build_dir/"main.pdf",pdf)
    files = ["main.tex","revision_numbers.tex","revision_results.tex","references.bib",
        "IEEEtran.cls","IEEEtran.bst","figures/caf_problem_solution.pdf",
        "figures/caf_overall_flow.pdf","figures/revision_main.pdf"]
    archive = paper/"CAF_arxiv_source.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(paper/name,name)
        z.write(args.build_dir/"main.bbl","main.bbl")
    (paper/"package-sha256.json").write_text(json.dumps(
        {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (pdf,archive)},indent=2)+"\n")
    print(pdf)
    print(archive)


if __name__ == "__main__":
    main()
