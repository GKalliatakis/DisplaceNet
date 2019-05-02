## Engine

Engine module contains the source code for the three branches of the DisplaceNet:

    1. Object detection branch - detects all humans in an image
    2. Human-centric branch - for each detected **human** we conduct continuous emotion recognition in VAD space
        and then we detect the overall dominance level that characterises the entire image
    3. Displaced people branch - We label the image as either _displaced people_ or _non displaced people_ based on **image classification** and **dominance level**




### Object detection branch
Contains the source code for two popular object detectors: RetinaNet and SSD.

### Human-centric branch
Contains the source code for conducting continuous emotion recognition in VAD space, for each detected human,
from their frame of reference & detecting the overall dominance level that characterises the entire image.

### Displaced people branch
Contains the source code for assigning a label to the input image based on image classification and overall dominance level.

```
