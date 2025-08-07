import click
from .config import load_config
from .extractor import run_extractor
from .scalar import run_scalar
from .ops.warp_label import warp_to_mni
from .ops.extract_roi_metrics import extract
from .ops.compute_r1_map import compute_r1

@click.group()
def cli():
    """func-struct extractor CLI"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help="YAML or JSON config for extractor pipeline")
def extractor(config):
    cfg = load_config(config)
    run_extractor(cfg)

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help="YAML or JSON config for scalar extraction")
def scalar(config):
    cfg = load_config(config)
    run_scalar(cfg)

@cli.command()
@click.option('--native-label', '-nl', required=True, type=click.Path(exists=True))
@click.option('--t1-native', '-t1', required=True, type=click.Path(exists=True))
@click.option('--mni-template', '-mt', required=True, type=click.Path(exists=True))
@click.option('--out', '-o', required=True, type=click.Path())
def warp_label(native_label, t1_native, mni_template, out):
    """Warp native-space label to MNI"""
    warp_to_mni(native_label, mni_template, out)
    click.echo(f"Wrote: {out}")

@cli.command()
@click.option('--subject', '-s', required=True)
@click.option('--labels', '-l', multiple=True, type=click.Path(exists=True))
@click.option('--vars', '-v', multiple=True, type=click.Path(exists=True))
@click.option('--var_names', '-n', multiple=True, required=True)
@click.option('--out_csv', '-o', required=True, type=click.Path())
def extract_metrics(subject, labels, vars, var_names, out_csv):
    """Extract mean metrics from labels"""
    df = extract(subject, labels, vars, var_names)
    df.to_csv(out_csv, index=False)
    click.echo(f"Saved: {out_csv}")

@cli.command()
@click.option('--t1', '-t1', required=True, type=click.Path(exists=True))
@click.option('--t2', '-t2', required=True, type=click.Path(exists=True))
@click.option('--out', '-o', required=True, type=click.Path())
def compute_r1_cmd(t1, t2, out):
    """Calculate T1/T2 ratio as proxy for R1"""
    compute_r1(t1, t2, out)
    click.echo(f"Saved R1 map: {out}")

if __name__ == "__main__":
    cli()

