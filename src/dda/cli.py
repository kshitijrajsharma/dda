import argparse
import json
import logging
import sys

from dda.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dda", description="DINOv3 siamese building damage assessment")
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", "-c", default="conf/train.yaml", help="Path to YAML config")
        p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. data_pct=10")

    _common(sub.add_parser("train", help="Train the damage model"))

    p_predict = sub.add_parser("predict", help="Damage assessment on a post GeoTIFF + building GeoJSON")
    p_predict.add_argument("--ckpt", default=None, help="Local checkpoint; omit to fetch from the Hub")
    p_predict.add_argument("--raster", required=True, help="Post-disaster GeoTIFF")
    p_predict.add_argument("--buildings", required=True, help="Building footprints GeoJSON")
    p_predict.add_argument("--pre-raster", default=None, help="Optional pre-disaster GeoTIFF")
    p_predict.add_argument("--out", required=True, help="Output GeoJSON with per-building damage")
    _common(p_predict)

    p_export = sub.add_parser("export", help="Export checkpoint to ONNX")
    p_export.add_argument("--ckpt", default=None, help="Local checkpoint; omit to fetch from the Hub")
    p_export.add_argument("--out", required=True)
    _common(p_export)

    p_eval = sub.add_parser("evaluate", help="Object-level (per-building) damage F1")
    p_eval.add_argument("--ckpt", default=None, help="Local checkpoint; omit to fetch from the Hub")
    p_eval.add_argument("--split", default="val", choices=["val", "test"])
    _common(p_eval)

    p_cal = sub.add_parser("calibrate", help="Fit confidence temperature on the val split")
    p_cal.add_argument("--ckpt", default=None, help="Local checkpoint; omit to fetch from the Hub")
    _common(p_cal)

    return parser


def app() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=list(args.overrides))

    if args.command == "train":
        from dda.train import train

        print(json.dumps(train(cfg), indent=2))
    elif args.command == "predict":
        from dda.infer import predict_damage, resolve_ckpt

        predict_damage(
            cfg,
            ckpt_path=resolve_ckpt(cfg, args.ckpt),
            post_raster=args.raster,
            buildings_geojson=args.buildings,
            out_geojson=args.out,
            pre_raster=args.pre_raster,
        )
    elif args.command == "export":
        from dda.export import export_onnx
        from dda.infer import resolve_ckpt

        export_onnx(cfg, ckpt_path=resolve_ckpt(cfg, args.ckpt), out_path=args.out)
    elif args.command == "evaluate":
        from dda.evaluation import object_level_eval
        from dda.infer import resolve_ckpt

        result = object_level_eval(cfg, ckpt_path=resolve_ckpt(cfg, args.ckpt), split=args.split)
        print(json.dumps(result, indent=2))
    elif args.command == "calibrate":
        from dda.calibrate import fit_temperature
        from dda.infer import resolve_ckpt

        temperature = fit_temperature(cfg, ckpt_path=resolve_ckpt(cfg, args.ckpt))
        print(json.dumps({"temperature": temperature}, indent=2))
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(app())
