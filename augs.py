import albumentations as A

def get_light_augmentations(img_sz):
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15,
                           border_mode= cv2.BORDER_CONSTANT, value=0),
        A.RandomSizedCrop(min_max_height=(int(img_sz[0] * 0.85), img_sz[0]),
                          height= image_size[0],
                          width = image_size[1], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(75, 125)),
            A.NoOp()]),
        A.OneOf([
            A.CLAHE(),
            A.NoOp()
        ]),
        A.HorizontalFlip(p=0.5),
    ])


def get_medium_augmentations(image_size):
    return A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0),
            A.NoOp()
        ]),
        A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
                          height=image_size[0],
                          width=image_size[1], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),
        A.OneOf([
            A.FancyPCA(alpha_std=4),
            A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=5),
            A.NoOp()
        ]),
        A.OneOf([
            A.CLAHE(),
            A.NoOp()
        ]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

def get_hard_augmentations(image_size):
    return A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=45,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(alpha_affine=0,
                               alpha=35,
                               sigma=5,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=0),
            A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT,
                             value=0),
            A.NoOp()
        ]),

        A.OneOf([
            ZeroTopAndBottom(p=0.3),

            A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
                              height=image_size[0],
                              width=image_size[1], p=0.3),
            A.NoOp()
        ]),

        A.ISONoise(p=0.5),

        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),

        A.OneOf([
            A.FancyPCA(alpha_std=6),
            A.RGBShift(r_shift_limit=40, b_shift_limit=30, g_shift_limit=30),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=10),
            A.ToGray(p=0.2),
            A.NoOp()
        ]),

        # Intentionally destroy image quality and assign 0 class in this case
        # A.Compose([
        #     BrightnessContrastDestroy(p=0.1),
        #     A.OneOf([
        #         MakeTooBlurry(),
        #         MakeTooBlurryMedian(),
        #         A.NoOp()
        #     ], p=0.1),
        # ]),

        ChannelIndependentCLAHE(p=0.5),

        A.ChannelDropout(),
        A.RandomGridShuffle(p=0.3),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ])
    ]
