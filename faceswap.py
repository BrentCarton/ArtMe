import warnings
warnings.filterwarnings("ignore")
from models import FaceTranslationGANInferenceModel

model = FaceTranslationGANInferenceModel()

from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
fv = FaceVerifier(classes=512)

from face_toolbox_keras.models.parser import face_parser
fp = face_parser.FaceParser()

from face_toolbox_keras.models.detector import face_detector
fd = face_detector.FaceAlignmentDetector()

from face_toolbox_keras.models.detector.iris_detector import IrisDetector
idet = IrisDetector()

import numpy as np
from utils import utils
from matplotlib import pyplot as plt

import cv2

def faceswap(painting):
    data =  cv2.imread(painting)
    fn_src={
        painting : data
    }

    fns_tar={
        "faces.jpg" : data
    }

    fn_src = [k for k,v in fn_src.items()]
    if len(fn_src) >= 1:
        fn_src = fn_src[0]

    fns_tar = [k for k,v in fns_tar.items()]

    print(fn_src)
    print(fns_tar)

    src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(fn_src, fd, fp, idet)
    tar, emb_tar = utils.get_tar_inputs(fns_tar, fd, fv)

    out = model.inference(src, mask, tar, emb_tar)

    fig = plt.figure()

    fig.set_size_inches(10,10)

    plt.axis('off')

    result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
    plt.imshow(result_face)
    result_img = utils.post_process_result(fn_src, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
    plt.imshow(result_img)
    plt.savefig("test.png", bbox_inches='tight')


