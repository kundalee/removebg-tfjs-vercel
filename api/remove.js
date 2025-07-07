import * as tf from '@tensorflow/tfjs';
import * as bodyPix from '@tensorflow-models/body-pix';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST allowed');
  }

  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).send('Missing imageBase64');

    // Initialize TensorFlow.js
    await tf.ready();

    // Load and decode image
    const imgTensor = tf.node.decodeImage(Buffer.from(imageBase64, 'base64'));
    
    // Load the model and perform segmentation
    const net = await bodyPix.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      multiplier: 0.75,
      quantBytes: 2
    });
    
    const segmentation = await net.segmentPerson(imgTensor, {
      flipHorizontal: false,
      internalResolution: 'medium',
      segmentationThreshold: 0.7
    });

    // Create output tensor
    const [height, width] = imgTensor.shape;
    const outputArray = new Uint8Array(width * height * 4);
    const inputArray = imgTensor.dataSync();

    for (let i = 0; i < segmentation.data.length; i++) {
      const isPerson = segmentation.data[i];
      const baseIdx = i * 4;
      const inputIdx = i * 3;
      
      outputArray[baseIdx] = inputArray[inputIdx];     // R
      outputArray[baseIdx + 1] = inputArray[inputIdx + 1]; // G
      outputArray[baseIdx + 2] = inputArray[inputIdx + 2]; // B
      outputArray[baseIdx + 3] = isPerson ? 255 : 0;      // A
    }

    // Convert to base64
    const buffer = Buffer.from(outputArray);
    const resultBase64 = buffer.toString('base64');

    // Cleanup
    tf.dispose([imgTensor]);

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error('Error:', e);
    res.status(500).send('Failed to remove background');
  }
}