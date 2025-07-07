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
    
    // Create image element
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = `data:image/jpeg;base64,${imageBase64}`;
    });

    // Create canvas and draw image
    const canvas = new OffscreenCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const tensor = tf.browser.fromPixels(imageData, 3);
    
    // Load the model and perform segmentation
    const net = await bodyPix.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      multiplier: 0.75,
      quantBytes: 2
    });
    
    const segmentation = await net.segmentPerson(tensor, {
      flipHorizontal: false,
      internalResolution: 'medium',
      segmentationThreshold: 0.7
    });

    // Create output image data
    const outputData = new Uint8ClampedArray(img.width * img.height * 4);
    const inputData = imageData.data;

    for (let i = 0; i < segmentation.data.length; i++) {
      const isPerson = segmentation.data[i];
      const baseIdx = i * 4;
      
      outputData[baseIdx] = inputData[baseIdx];       // R
      outputData[baseIdx + 1] = inputData[baseIdx + 1]; // G
      outputData[baseIdx + 2] = inputData[baseIdx + 2]; // B
      outputData[baseIdx + 3] = isPerson ? 255 : 0;     // A
    }

    // Create new image data and put it on canvas
    const outputImageData = new ImageData(outputData, img.width, img.height);
    ctx.putImageData(outputImageData, 0, 0);

    // Convert to blob and then to base64
    const blob = await canvas.convertToBlob();
    const resultBuffer = await blob.arrayBuffer();
    const resultBase64 = Buffer.from(resultBuffer).toString('base64');

    // Cleanup
    tf.dispose(tensor);

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error('Error:', e.stack);
    res.status(500).send(`Failed to remove background: ${e.message}`);
  }
}