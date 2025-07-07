import * as tf from '@tensorflow/tfjs';
import * as bodyPix from '@tensorflow-models/body-pix';
import sharp from 'sharp';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST allowed');
  }

  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).send('Missing imageBase64');

    // Initialize TensorFlow.js
    await tf.ready();
    
    // Convert base64 to buffer
    const inputBuffer = Buffer.from(imageBase64, 'base64');
    
    // Process image with sharp
    const { data, info } = await sharp(inputBuffer)
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Convert to tensor
    const tensor = tf.tensor3d(new Uint8Array(data), [info.height, info.width, info.channels]);
    
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

    // Create output buffer
    const outputData = new Uint8Array(info.width * info.height * 4);
    const inputData = new Uint8Array(data);

    for (let i = 0; i < segmentation.data.length; i++) {
      const isPerson = segmentation.data[i];
      const inputIdx = i * info.channels;
      const outputIdx = i * 4;
      
      outputData[outputIdx] = inputData[inputIdx];     // R
      outputData[outputIdx + 1] = inputData[inputIdx + 1]; // G
      outputData[outputIdx + 2] = inputData[inputIdx + 2]; // B
      outputData[outputIdx + 3] = isPerson ? 255 : 0;     // A
    }

    // Convert to PNG with transparency
    const resultBuffer = await sharp(outputData, {
      raw: {
        width: info.width,
        height: info.height,
        channels: 4
      }
    })
      .png()
      .toBuffer();

    // Convert to base64
    const resultBase64 = resultBuffer.toString('base64');

    // Cleanup
    tf.dispose(tensor);

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error('Error:', e.stack);
    res.status(500).send(`Failed to remove background: ${e.message}`);
  }
}