import * as tf from '@tensorflow/tfjs';
import * as bodyPix from '@tensorflow-models/body-pix';
import { createCanvas, Image, loadImage } from 'canvas';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST allowed');
  }

  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).send('Missing imageBase64');

    // Initialize TensorFlow.js
    await tf.ready();
    
    // Load image using canvas
    const buffer = Buffer.from(imageBase64, 'base64');
    const img = await loadImage(buffer);

    // Create canvas and draw image
    const canvas = createCanvas(img.width, img.height);
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
    const outputData = ctx.createImageData(img.width, img.height);
    const inputData = imageData.data;

    for (let i = 0; i < segmentation.data.length; i++) {
      const isPerson = segmentation.data[i];
      const baseIdx = i * 4;
      
      outputData.data[baseIdx] = inputData[baseIdx];       // R
      outputData.data[baseIdx + 1] = inputData[baseIdx + 1]; // G
      outputData.data[baseIdx + 2] = inputData[baseIdx + 2]; // B
      outputData.data[baseIdx + 3] = isPerson ? 255 : 0;     // A
    }

    // Put the processed image data back to canvas
    ctx.putImageData(outputData, 0, 0);

    // Convert to base64
    const resultBase64 = canvas.toBuffer().toString('base64');

    // Cleanup
    tf.dispose(tensor);

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error('Error:', e.stack);
    res.status(500).send(`Failed to remove background: ${e.message}`);
  }
}