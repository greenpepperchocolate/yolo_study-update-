"""
Form parameters for training and augmentation, with their limits.

One definition drives both the HTML form (labels, min/max/step) and the
server-side validation, so the two can never disagree.
"""


class ParamError(ValueError):
    """Raised when a submitted value is missing or out of range."""


class Param:
    """A numeric form field with inclusive bounds."""

    def __init__(self, key, label, kind, default, minimum, maximum,
                 step=None, help_text='', unit=''):
        self.key = key
        self.label = label
        self.kind = kind
        self.default = default
        self.min = minimum
        self.max = maximum
        self.step = step or (1 if kind is int else 0.01)
        self.help = help_text
        self.unit = unit

    def parse(self, raw):
        """Convert a submitted string, enforcing type and bounds."""
        if raw is None or str(raw).strip() == '':
            raise ParamError(f'「{self.label}」を入力してください')
        try:
            value = float(raw)
        except ValueError:
            raise ParamError(f'「{self.label}」は数値で入力してください') \
                from None
        if self.kind is int:
            if value != int(value):
                raise ParamError(f'「{self.label}」は整数で入力してください')
            value = int(value)
        if not self.min <= value <= self.max:
            raise ParamError(
                f'「{self.label}」は {self.min}〜{self.max}'
                f'{self.unit} の範囲で入力してください')
        return value


MODELS = [
    ('yolo26n.pt', 'YOLO26n（最軽量・CPU 向け）'),
    ('yolo26s.pt', 'YOLO26s（軽量）'),
    ('yolo26m.pt', 'YOLO26m（中）'),
    ('yolo26l.pt', 'YOLO26l（大・GPU 推奨）'),
    ('yolo26x.pt', 'YOLO26x（最大・GPU 推奨）'),
]
MODEL_KEYS = {m for m, _ in MODELS}

TRAIN_PARAMS = [
    Param('epochs', 'エポック数', int, 50, 1, 1000,
          help_text='学習を繰り返す回数。少ないデータなら 50〜100 が目安'),
    Param('imgsz', '画像サイズ', int, 640, 320, 1280, step=32, unit='px',
          help_text='32 の倍数。大きいほど小さな物体に強いが遅くなる'),
    Param('batch', 'バッチサイズ', int, 8, 1, 64,
          help_text='一度に処理する枚数。メモリ不足で落ちるなら下げる'),
    Param('patience', '早期終了', int, 30, 1, 1000, unit=' エポック',
          help_text='検証スコアがこの回数改善しなければ打ち切る'),
    Param('val_percent', '検証用の割合', float, 20, 5, 50, step=1,
          unit='%', help_text='自動分割するとき、評価に回す画像の割合'),
]

# Ultralytics' built-in augmentation, applied on the fly during training.
ONLINE_AUG_PARAMS = [
    Param('fliplr', '左右反転の確率', float, 0.5, 0, 1),
    Param('flipud', '上下反転の確率', float, 0.0, 0, 1),
    Param('degrees', '回転', float, 0.0, 0, 180, step=1, unit='°',
          help_text='±この角度の範囲でランダムに回転'),
    Param('translate', '平行移動', float, 0.1, 0, 0.9,
          help_text='画像サイズに対する割合'),
    Param('scale', '拡大縮小', float, 0.5, 0, 0.9,
          help_text='±この割合でランダムに拡大・縮小'),
    Param('shear', 'せん断（斜め変形）', float, 0.0, 0, 45, step=1,
          unit='°'),
    Param('hsv_h', '色相のゆらぎ', float, 0.015, 0, 1, step=0.005),
    Param('hsv_s', '彩度のゆらぎ', float, 0.7, 0, 1),
    Param('hsv_v', '明度のゆらぎ', float, 0.4, 0, 1),
    Param('mosaic', 'モザイク（4 枚合成）の確率', float, 1.0, 0, 1,
          help_text='小さな物体や背景の多様化に効く'),
    Param('mixup', 'MixUp（2 枚重ね）の確率', float, 0.0, 0, 1),
]

ONLINE_AUG_PRESETS = {
    'standard': {p.key: p.default for p in ONLINE_AUG_PARAMS},
    'light': {'fliplr': 0.5, 'flipud': 0.0, 'degrees': 0, 'translate': 0.05,
              'scale': 0.2, 'shear': 0, 'hsv_h': 0.01, 'hsv_s': 0.3,
              'hsv_v': 0.2, 'mosaic': 0.5, 'mixup': 0.0},
    'strong': {'fliplr': 0.5, 'flipud': 0.2, 'degrees': 15,
               'translate': 0.2, 'scale': 0.7, 'shear': 5, 'hsv_h': 0.03,
               'hsv_s': 0.8, 'hsv_v': 0.5, 'mosaic': 1.0, 'mixup': 0.2},
    'none': {p.key: 0 for p in ONLINE_AUG_PARAMS},
}


class OfflineTransform:
    """One pre-generated augmentation: a probability plus options."""

    def __init__(self, key, label, default_percent, options=(),
                 help_text=''):
        self.key = key
        self.label = label
        self.prob = Param(f'{key}_p', f'{label}の確率', float,
                          default_percent, 0, 100, step=1, unit='%')
        self.options = list(options)
        self.help = help_text


OFFLINE_COPIES = Param(
    'copies', '1 枚あたりの水増し枚数', int, 0, 0, 10, unit=' 枚',
    help_text='0 で事前の水増しをしない。学習用画像だけが対象')
MAX_OFFLINE_IMAGES = 20000

OFFLINE_TRANSFORMS = [
    OfflineTransform('hflip', '左右反転', 50),
    OfflineTransform('vflip', '上下反転', 0),
    OfflineTransform('rotate', '回転', 30, [
        Param('rotate_limit', '回転の最大角度', int, 15, 1, 90, unit='°')]),
    OfflineTransform('scale', '拡大縮小', 0, [
        Param('scale_range', '拡大縮小の幅', float, 0.2, 0.05, 0.5,
              help_text='0.2 なら 0.8〜1.2 倍')]),
    OfflineTransform('brightness', '明るさ・コントラスト', 30, [
        Param('brightness_limit', '変化の強さ', float, 0.2, 0.05, 0.5)]),
    OfflineTransform('hsv', '色相・彩度', 20, [
        Param('hue_shift', '色相のずらし幅', int, 10, 1, 90),
        Param('sat_shift', '彩度のずらし幅', int, 30, 1, 100)]),
    OfflineTransform('gray', 'グレースケール', 20, help_text=(
        '白黒カメラで使う場合や、色に頼らず形で覚えさせたいときに。'
        '100% にすると水増し画像がすべて白黒になる')),
    OfflineTransform('blur', 'ぼかし', 10, [
        Param('blur_max', 'ぼかしの最大サイズ', int, 5, 3, 15, unit='px',
              help_text='奇数に丸める')]),
    OfflineTransform('noise', 'ノイズ', 10, [
        Param('noise_std', 'ノイズの強さ', float, 0.05, 0.01, 0.3)]),
    OfflineTransform('jpeg', 'JPEG 圧縮の劣化', 0, [
        Param('jpeg_quality', '最低画質', int, 40, 10, 95)]),
    OfflineTransform('cutout', '一部の塗りつぶし', 0, [
        Param('cutout_holes', '塗りつぶす最大個数', int, 4, 1, 20)],
        help_text='物体が一部隠れても検出できるようにする'),
]


PREDICT_PARAMS = [
    Param('conf', '信頼度のしきい値', float, 0.25, 0.01, 1,
          help_text='これより低い確信度の検出は表示しない'),
    Param('imgsz', '画像サイズ', int, 640, 320, 1280, step=32, unit='px',
          help_text='学習時と同じにするのが基本'),
]

EXPORT_FORMATS = [
    ('onnx', 'ONNX（汎用。ONNX Runtime や他言語から使う）'),
    ('ncnn', 'NCNN（スマホ・Raspberry Pi などの組み込み向け）'),
    ('torchscript', 'TorchScript（PyTorch / C++ から使う）'),
]
EXPORT_KEYS = {k for k, _ in EXPORT_FORMATS}


def all_offline_params():
    """Every Param used by the offline augmentation form."""
    params = [OFFLINE_COPIES]
    for t in OFFLINE_TRANSFORMS:
        params.append(t.prob)
        params.extend(t.options)
    return params


def parse_group(form, params, prefix=''):
    """Parse a list of Params from a form; collect every error."""
    values, errors = {}, []
    for p in params:
        try:
            values[p.key] = p.parse(form.get(prefix + p.key))
        except ParamError as e:
            errors.append(str(e))
    return values, errors


def defaults(params):
    """Default values for a list of Params."""
    return {p.key: p.default for p in params}
