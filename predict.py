import hashlib
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from tqdm import tqdm

import models

SR = 16000
MEL_ARGS = {
        'n_fft' : 2048,
        'n_mels' : 64,
        'hop_length' : 320,
        'win_length' : 640,
        'sr': SR
    }

ABBREV_TO_FULLNAME = {
    "ab": "Abkhazian",
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gn": "Guarani",
    "gu": "Gujarati",
    "gv": "Manx",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "ia": "Interlingua",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "iw": "Hebrew",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nyorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Panjabi",
    "pl": "Polish",
    "ps": "Pushto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sco": "Scots",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "war": "Waray",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
}


MODELS = {
    'CNN10': {
        'url': 'https://zenodo.org/record/4436037/files/CNN10.pth?download=1',
        'md5': 'ca56e5003b5025eff6f991e47ba87b06'
    },
    'CNN6': {
        'url': 'https://zenodo.org/record/4436037/files/CNN6.pth?download=1',
        'md5': 'b0ae5a1bce63fa5522939fa123d3f0a3',
    },
    'CNNVAD':{
        'url': 'https://zenodo.org/record/4436037/files/CNNVAD.pth?download=1',
        'md5' : '4785073a07a61c4f431e85a14da9aca1',
        },
    'MobileNetV2':
    {'url':
    'https://zenodo.org/record/4436037/files/MobileNetV2.pth?download=1',
    'md5':'29f3903813610dfc779d9d26875e6929'
    }
}



def predict_language(audiofilepath: Path, model='CNN6', N:int=5, return_results=False):
    assert model in MODELS.keys(), f"Possible choice for models are {list(MODELS.keys())}"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPS = np.spacing(1)
    def _extract_feature(path):
        y, sr = sf.read(path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(1)
        y = librosa.resample(y, sr, SR)
        return np.log(librosa.feature.melspectrogram(y, **MEL_ARGS) + EPS).T

    feature = torch.as_tensor(
        _extract_feature(audiofilepath)).float().to(DEVICE)
    try:
        model_params = _download_model(model)
    except ValueError:
        print("Download has failed, aborting!")
        return
    model = getattr(models, model)(inputdim=64,
                                   outputdim=len(ABBREV_TO_FULLNAME))
    model.load_state_dict(model_params)
    model = model.to(DEVICE).eval()
    # pd.options.display.float_format = '{:,.2f}'.format
    with torch.no_grad():
        y = torch.softmax(model(feature.unsqueeze(0)), dim=-1).to('cpu').squeeze(0).numpy()
    label_names = list(ABBREV_TO_FULLNAME.values())
    print(f"Top-{N} results")
    idxs  = np.argsort(y)[::-1]
    for idx in idxs[:N]:
        print(f"{y[idx]:.2f} {label_names[idx]}")
    if return_results:
        return y


def _download_with_progress_bar(url, file_name, md5=None):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(file_name, 'wb') as file, tqdm(total=total_size_in_bytes,
                                            unit='iB',
                                            unit_scale=True) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    if (total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes):
        raise ValueError("Download has not been sucessful!")
    if md5:
        with open(file_name, 'rb') as r:
            if hashlib.md5(r.read()).hexdigest() != md5:
                raise ValueError("Download not sucessful, file is corrupted.")

        



def _download_model(modelname, pretrained_model_dir='pretrained_models'):
    url = MODELS[modelname]['url']
    md5 = MODELS[modelname]['md5']
    print(f"Downloading from {url}")
    Path(pretrained_model_dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{pretrained_model_dir}/{modelname}.pth'

    
    if Path(file_name).exists():
        with open(file_name,'rb') as r:
            if hashlib.md5(r.read()).hexdigest() != md5:
                _download_with_progress_bar(url, file_name)
        print(f"Found model {file_name}")
        return torch.load(file_name,map_location='cpu')
    else:
        _download_with_progress_bar(url, file_name)
    return torch.load(file_name,map_location='cpu')

if __name__ == "__main__":
    # _download_model('CNN6')
    import fire
    fire.Fire(predict_language)
