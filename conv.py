path = 'vtuber.h5'

import coremltools
coreml_model = coremltools.converters.keras.convert(path,
        input_names = 'image',
        image_input_names = 'image',
        #is_bgr = True,
        class_labels = ['kizunaai', 'miraiakari', 'kaguyaruna'],)

coreml_model.save('vtuber.mlmodel')
