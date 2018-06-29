from PIL import Image, ImageEnhance, ImageFilter

im = Image.open("VAL_data/Bosch_GluePen_Akku-Heißklebepistole/578295_600_370.png")

im.show()

enhancer_contrast = ImageEnhance.Contrast(im)
enhancer_brightness = ImageEnhance.Brightness(im)
enhancer_color = ImageEnhance.Color(im)

im = im.filter(ImageFilter.GaussianBlur(3)).show()

enhancer_brightness.enhance(0.5).show("XXX")
enhancer_color.enhance(0.5).show("YYY")
enhancer_contrast.enhance(0.5).show("ZzZ")
