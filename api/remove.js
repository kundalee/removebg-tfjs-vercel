import * as tf from '@tensorflow/tfjs';
import * as bodyPix from '@tensorflow-models/body-pix';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST allowed');
  }

  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).send('Missing imageBase64');

    // Create an off-screen canvas
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = `data:image/png;base64,${imageBase64}`;
    });

    const canvas = new OffscreenCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Load the model and perform segmentation
    const net = await bodyPix.load({ architecture: 'MobileNetV1', outputStride: 16 });
    const segmentation = await net.segmentPerson(canvas);

    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const newImageData = new ImageData(
      new Uint8ClampedArray(imageData.data.length),
      imageData.width,
      imageData.height
    );

    for (let i = 0; i < segmentation.data.length; i++) {
      const isPerson = segmentation.data[i] === 1;
      newImageData.data[i * 4 + 0] = imageData.data[i * 4 + 0];
      newImageData.data[i * 4 + 1] = imageData.data[i * 4 + 1];
      newImageData.data[i * 4 + 2] = imageData.data[i * 4 + 2];
      newImageData.data[i * 4 + 3] = isPerson ? 255 : 0;
    }

    ctx.putImageData(newImageData, 0, 0);
    const blob = await canvas.convertToBlob();
    const resultBuffer = await blob.arrayBuffer();
    const resultBase64 = Buffer.from(resultBuffer).toString('base64');

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error(e);
    res.status(500).send('Failed to remove background');
  }
}