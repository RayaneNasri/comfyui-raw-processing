from torchvision.io import decode_image

def read_image(image_name: str):
    img = decode_image(image_name)   # uint8, C x H x W
    img = img.float() / 255.0      # float32 dans [0,1]
    img = img.permute(1,2,0)
    return img