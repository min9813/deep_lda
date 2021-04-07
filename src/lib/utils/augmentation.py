import cv2
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
import imgaug as ia
from imgaug import augmenters as iaa
from random import randint, uniform


class EzImageBaseAug(object):
    def __init__(self, size=384, random_erase_p=0.0):
        def very_rare(aug): return iaa.Sometimes(0.1, aug)
        def rare(aug): return iaa.Sometimes(0.3, aug)
        def sometimes(aug): return iaa.Sometimes(0.5, aug)
        def often(aug): return iaa.Sometimes(0.8, aug)
        self.random_erase_p = random_erase_p
        self.seq = iaa.Sequential(
            [
                rare(iaa.HorizontalFlip()),
                # iaa.Affine(
                #     #scale={"x": (.5,.5), "y": (.5,.5)}, # scale images to 80-120% of their size, individually per axis
                #     scale={"x": (0.8,1.0), "y": (0.8, 1.0)}, # scale images to 80-120% of their size, individually per axis
                #     #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to 20 percent (per axis)
                #     rotate=(-5, 5), # rotate by -45 to 45 degrees
                #     shear=(-16, 16), # shear by -16 to 16 degrees
                #     order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #     cval=0, # if mode is constant, use a cval between 0 and 255
                #     mode="constant", # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # ),
            ],
            random_order=True
        )
        self.size = size

    def __call__(self, img, points):

        if len(points):
            points_ = []
            for point in points:
                x, y = point
                points_.append(
                    ia.Keypoint(x=x, y=y)
                )
            keypoints = ia.KeypointsOnImage(
                points_, shape=img.shape)

        seq_det = self.seq.to_deterministic()
        img_aug = seq_det.augment_images(
                        [img]
                    )[0]

        if len(points):
            keypoints_aug = seq_det.augment_keypoints(
                            [keypoints]
                    )[0]
            for i in range(len(keypoints_aug.keypoints)):
                p = keypoints_aug.keypoints[i]
    #             x,y = points[i]
                points[i] = [p.x, p.y]

        return img_aug, points


def horizontal_flip(image, pose=[], p=0.5):
    if np.random.random() <= p:
        image = image[:, ::-1].copy()
        if len(pose):
            h, w, c = image.shape
            pose[:, 0] = w - pose[:, 0]

    return image, pose


class EzImageBaseAugAffine(object):
    def __init__(self, size=384, random_erase_p=0.0):
        def very_rare(aug): return iaa.Sometimes(0.1, aug)
        def rare(aug): return iaa.Sometimes(0.3, aug)
        def sometimes(aug): return iaa.Sometimes(0.5, aug)
        def often(aug): return iaa.Sometimes(0.8, aug)
        self.random_erase_p = random_erase_p
        self.seq = iaa.Sequential(
            [
                rare(iaa.HorizontalFlip()),
                rare(iaa.Affine(
                    # scale={"x": (.5,.5), "y": (.5,.5)}, # scale images to 80-120% of their size, individually per axis
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    # translate by -20 to 20 percent (per axis)
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),  # rotate by -45 to 45 degrees
                    shear=(-5, 5),  # shear by -16 to 16 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    cval=0,  # if mode is constant, use a cval between 0 and 255
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode="constant",
                ))]
        )
        self.size = size

    def __call__(self, img, points):

        if len(points):
            points_ = []
            for point in points:
                x, y = point
                points_.append(
                    ia.Keypoint(x=x, y=y)
                )
            keypoints = ia.KeypointsOnImage(
                points_, shape=img.shape)

        seq_det = self.seq.to_deterministic()
        img_aug = seq_det.augment_images(
                        [img]
                    )[0]

        if len(points):
            keypoints_aug = seq_det.augment_keypoints(
                            [keypoints]
                    )[0]
            for i in range(len(keypoints_aug.keypoints)):
                p = keypoints_aug.keypoints[i]
    #             x,y = points[i]
                points[i] = [p.x, p.y]

        return img_aug, points


def get_strong_shape_aug():
    s_aug = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=[-0.3, 0.5], rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]
    )

    return s_aug


def get_aug_trans(use_color_aug, use_weak_shape_aug, use_strong_shape_aug, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # range [0.0, 1.0] -> [-1.0,1.0]
        transforms.Normalize(mean=mean, std=std)
    ])

    if use_color_aug:
        c_aug = A.Compose([
            A.RandomBrightnessContrast(
                p=0.7, brightness_limit=0.2, contrast_limit=0.2),
            A.CoarseDropout(p=0.5, max_holes=8, max_height=16,
                            max_width=16, min_height=8, min_width=8, fill_value=0),
            A.OneOf([
                    A.Blur(p=1, blur_limit=7),
                    A.MotionBlur(p=1, blur_limit=7),
                    A.MedianBlur(p=1, blur_limit=7),
                    A.GaussianBlur(p=1, blur_limit=7)
                    ], p=0.5),
            A.OneOf([
                    A.RandomGamma(p=1, gamma_limit=(80, 120)),
                    A.GaussNoise(p=1, var_limit=(10.0, 50.0)),
                    A.ISONoise(p=1, color_shift=(0.01, 0.05),
                               intensity=(0.1, 0.5)),
                    ], p=0.3),
            A.OneOf([
                    A.HueSaturationValue(p=1, hue_shift_limit=20,
                                         sat_shift_limit=30, val_shift_limit=20),
                    A.RGBShift(p=1, r_shift_limit=20,
                               g_shift_limit=20, b_shift_limit=20),
                    A.CLAHE(p=1, clip_limit=4.0, tile_grid_size=(8, 8)),
                    # A.ChannelShuffle(p=0.5),
                    # A.InvertImg(p=0.5),
                    A.Solarize(p=1, threshold=128),
                    ], p=0.5),
            A.JpegCompression(quality_lower=10, quality_upper=30, p=0.3),
        ])
    else:
        c_aug = None

    # crop_size = (args.train_input_h, args.train_input_w)

    if use_strong_shape_aug:
        shape_aug = get_strong_shape_aug()
    elif use_weak_shape_aug:
        shape_aug = horizontal_flip
    else:
        shape_aug = None

    return transform, c_aug, shape_aug


def get_unlabel_aug(use_color_aug, use_shape_aug):
    if use_color_aug:
        c_aug = A.Compose([
            A.RandomBrightnessContrast(
                p=0.8, brightness_limit=0.5, contrast_limit=0.5),
            A.CoarseDropout(p=0.8, max_holes=16, max_height=20,
                            max_width=20, min_height=8, min_width=8, fill_value=0),
            A.OneOf([
                    A.Blur(p=1, blur_limit=7),
                    A.MotionBlur(p=1, blur_limit=7),
                    A.MedianBlur(p=1, blur_limit=7),
                    A.GaussianBlur(p=1, blur_limit=7)
                    ], p=0.8),
            A.OneOf([
                    A.RandomGamma(p=1, gamma_limit=(80, 120)),
                    A.GaussNoise(p=1, var_limit=(10.0, 50.0)),
                    A.ISONoise(p=1, color_shift=(0.1, 0.5),
                               intensity=(0.3, 1.0)),
                    ], p=0.8),
            # A.CLAHE(p=0.7, clip_limit=4.0, tile_grid_size=(8, 8)),

            A.JpegCompression(quality_lower=10, quality_upper=30, p=0.8),
        ])
    else:
        c_aug = None

    # crop_size = (args.train_input_h, args.train_input_w)

    if use_shape_aug:
        shape_aug = EzImageBaseAug()
    else:
        shape_aug = None

    return c_aug, shape_aug


def get_gray_aug_trans(use_color_aug, use_shape_aug, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # range [0.0, 1.0] -> [-1.0,1.0]
        transforms.Normalize(mean=mean, std=std)
    ])

    if use_color_aug:
        c_aug = A.Compose([
            A.RandomBrightnessContrast(
                p=0.7, brightness_limit=0.5, contrast_limit=0.5),
            A.CoarseDropout(p=0.5, max_holes=8, max_height=16,
                            max_width=16, min_height=8, min_width=8, fill_value=0),
            A.OneOf([
                    A.Blur(p=1, blur_limit=7),
                    A.MotionBlur(p=1, blur_limit=7),
                    A.MedianBlur(p=1, blur_limit=7),
                    A.GaussianBlur(p=1, blur_limit=7)
                    ], p=0.5),
            A.OneOf([
                    A.RandomGamma(p=1, gamma_limit=(80, 120)),
                    A.GaussNoise(p=1, var_limit=(10.0, 50.0)),
                    A.ISONoise(p=1, color_shift=(0.01, 0.05),
                               intensity=(0.1, 0.5)),
                    ], p=0.3),
            A.JpegCompression(quality_lower=10, quality_upper=30, p=0.3),
            # A.JpegCompression(quality_lower=50, quality_upper=100, p=1),
        ])
    else:
        c_aug = None

    # crop_size = (args.train_input_h, args.train_input_w)

    if use_shape_aug:
        shape_aug = EzImageBaseAug()
    else:
        shape_aug = None

    return transform, c_aug, shape_aug



class EzImageBaseAug(object):
    def __init__(self, size=384, random_erase_p=0.0):
        def very_rare(aug): return iaa.Sometimes(0.1, aug)
        def rare(aug): return iaa.Sometimes(0.3, aug)
        def sometimes(aug): return iaa.Sometimes(0.5, aug)
        def often(aug): return iaa.Sometimes(0.8, aug)
        self.random_erase_p = random_erase_p
        self.seq = iaa.Sequential(
            [
                rare(iaa.HorizontalFlip()),
                # iaa.Affine(
                #     #scale={"x": (.5,.5), "y": (.5,.5)}, # scale images to 80-120% of their size, individually per axis
                #     scale={"x": (0.8,1.0), "y": (0.8, 1.0)}, # scale images to 80-120% of their size, individually per axis
                #     #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to 20 percent (per axis)
                #     rotate=(-5, 5), # rotate by -45 to 45 degrees
                #     shear=(-16, 16), # shear by -16 to 16 degrees
                #     order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #     cval=0, # if mode is constant, use a cval between 0 and 255
                #     mode="constant", # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # ),
            ],
            random_order=True
        )
        self.size = size

    def __call__(self, img, points):

        if len(points):
            points_ = []
            for point in points:
                x, y = point
                points_.append(
                    ia.Keypoint(x=x, y=y)
                )
            keypoints = ia.KeypointsOnImage(
                points_, shape=img.shape)

        seq_det = self.seq.to_deterministic()
        img_aug = seq_det.augment_images(
                        [img]
                    )[0]

        if len(points):
            keypoints_aug = seq_det.augment_keypoints(
                            [keypoints]
                    )[0]
            for i in range(len(keypoints_aug.keypoints)):
                p = keypoints_aug.keypoints[i]
    #             x,y = points[i]
                points[i] = [p.x, p.y]

        return img_aug, points


def horizontal_flip(image, pose=[], p=0.5):
    if np.random.random() <= p:
        image = image[:, ::-1].copy()
        if len(pose):
            h, w, c = image.shape
            pose[:, 0] = w - pose[:, 0]

    return image, pose


class EzImageBaseAugAffine(object):
    def __init__(self, size=384, random_erase_p=0.0):
        def very_rare(aug): return iaa.Sometimes(0.1, aug)
        def rare(aug): return iaa.Sometimes(0.3, aug)
        def sometimes(aug): return iaa.Sometimes(0.5, aug)
        def often(aug): return iaa.Sometimes(0.8, aug)
        self.random_erase_p = random_erase_p
        self.seq = iaa.Sequential(
            [
                rare(iaa.HorizontalFlip()),
                rare(iaa.Affine(
                    # scale={"x": (.5,.5), "y": (.5,.5)}, # scale images to 80-120% of their size, individually per axis
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    # translate by -20 to 20 percent (per axis)
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),  # rotate by -45 to 45 degrees
                    shear=(-5, 5),  # shear by -16 to 16 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    cval=0,  # if mode is constant, use a cval between 0 and 255
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode="constant",
                ))]
        )
        self.size = size

    def __call__(self, img, points):

        if len(points):
            points_ = []
            for point in points:
                x, y = point
                points_.append(
                    ia.Keypoint(x=x, y=y)
                )
            keypoints = ia.KeypointsOnImage(
                points_, shape=img.shape)

        seq_det = self.seq.to_deterministic()
        img_aug = seq_det.augment_images(
                        [img]
                    )[0]

        if len(points):
            keypoints_aug = seq_det.augment_keypoints(
                            [keypoints]
                    )[0]
            for i in range(len(keypoints_aug.keypoints)):
                p = keypoints_aug.keypoints[i]
    #             x,y = points[i]
                points[i] = [p.x, p.y]

        return img_aug, points


def get_strong_shape_aug():
    s_aug = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=[-0.3, 0.5], rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]
    )

    return s_aug


def get_aug_trans(use_color_aug, use_weak_shape_aug, use_strong_shape_aug, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # range [0.0, 1.0] -> [-1.0,1.0]
        transforms.Normalize(mean=mean, std=std)
    ])

    if use_color_aug:
        c_aug = A.Compose([
            A.RandomBrightnessContrast(
                p=0.7, brightness_limit=0.2, contrast_limit=0.2),
            A.CoarseDropout(p=0.5, max_holes=8, max_height=16,
                            max_width=16, min_height=8, min_width=8, fill_value=0),
            A.OneOf([
                    A.Blur(p=1, blur_limit=7),
                    A.MotionBlur(p=1, blur_limit=7),
                    A.MedianBlur(p=1, blur_limit=7),
                    A.GaussianBlur(p=1, blur_limit=7)
                    ], p=0.5),
            A.OneOf([
                    A.RandomGamma(p=1, gamma_limit=(80, 120)),
                    A.GaussNoise(p=1, var_limit=(10.0, 50.0)),
                    A.ISONoise(p=1, color_shift=(0.01, 0.05),
                               intensity=(0.1, 0.5)),
                    ], p=0.3),
            A.OneOf([
                    A.HueSaturationValue(p=1, hue_shift_limit=20,
                                         sat_shift_limit=30, val_shift_limit=20),
                    A.RGBShift(p=1, r_shift_limit=20,
                               g_shift_limit=20, b_shift_limit=20),
                    A.CLAHE(p=1, clip_limit=4.0, tile_grid_size=(8, 8)),
                    # A.ChannelShuffle(p=0.5),
                    # A.InvertImg(p=0.5),
                    A.Solarize(p=1, threshold=128),
                    ], p=0.5),
            A.JpegCompression(quality_lower=10, quality_upper=30, p=0.3),
        ])
    else:
        c_aug = None

    # crop_size = (args.train_input_h, args.train_input_w)

    if use_strong_shape_aug:
        shape_aug = get_strong_shape_aug()
    elif use_weak_shape_aug:
        shape_aug = horizontal_flip
    else:
        shape_aug = None

    return transform, c_aug, shape_aug


def get_unlabel_aug(use_color_aug, use_shape_aug):
    if use_color_aug:
        c_aug = A.Compose([
            A.RandomBrightnessContrast(
                p=0.8, brightness_limit=0.5, contrast_limit=0.5),
            A.CoarseDropout(p=0.8, max_holes=16, max_height=20,
                            max_width=20, min_height=8, min_width=8, fill_value=0),
            A.OneOf([
                    A.Blur(p=1, blur_limit=7),
                    A.MotionBlur(p=1, blur_limit=7),
                    A.MedianBlur(p=1, blur_limit=7),
                    A.GaussianBlur(p=1, blur_limit=7)
                    ], p=0.8),
            A.OneOf([
                    A.RandomGamma(p=1, gamma_limit=(80, 120)),
                    A.GaussNoise(p=1, var_limit=(10.0, 50.0)),
                    A.ISONoise(p=1, color_shift=(0.1, 0.5),
                               intensity=(0.3, 1.0)),
                    ], p=0.8),
            # A.CLAHE(p=0.7, clip_limit=4.0, tile_grid_size=(8, 8)),

            A.JpegCompression(quality_lower=10, quality_upper=30, p=0.8),
        ])
    else:
        c_aug = None

    # crop_size = (args.train_input_h, args.train_input_w)

    if use_shape_aug:
        shape_aug = EzImageBaseAug()
    else:
        shape_aug = None

    return c_aug, shape_aug


def get_gray_aug_trans(use_color_aug, use_shape_aug, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # range [0.0, 1.0] -> [-1.0,1.0]
        transforms.Normalize(mean=mean, std=std)
    ])

    if use_color_aug:
        c_aug = A.Compose([
            A.RandomBrightnessContrast(
                p=0.7, brightness_limit=0.5, contrast_limit=0.5),
            A.CoarseDropout(p=0.5, max_holes=8, max_height=16,
                            max_width=16, min_height=8, min_width=8, fill_value=0),
            A.OneOf([
                    A.Blur(p=1, blur_limit=7),
                    A.MotionBlur(p=1, blur_limit=7),
                    A.MedianBlur(p=1, blur_limit=7),
                    A.GaussianBlur(p=1, blur_limit=7)
                    ], p=0.5),
            A.OneOf([
                    A.RandomGamma(p=1, gamma_limit=(80, 120)),
                    A.GaussNoise(p=1, var_limit=(10.0, 50.0)),
                    A.ISONoise(p=1, color_shift=(0.01, 0.05),
                               intensity=(0.1, 0.5)),
                    ], p=0.3),
            A.JpegCompression(quality_lower=10, quality_upper=30, p=0.3),
            # A.JpegCompression(quality_lower=50, quality_upper=100, p=1),
        ])
    else:
        c_aug = None

    # crop_size = (args.train_input_h, args.train_input_w)

    if use_shape_aug:
        shape_aug = EzImageBaseAug()
    else:
        shape_aug = None

    return transform, c_aug, shape_aug