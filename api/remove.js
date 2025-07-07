import * as tf from '@tensorflow/tfjs-node';
import * as bodyPix from '@tensorflow-models/body-pix';
import { createCanvas, Image } from 'canvas';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST allowed');
  }

  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).send('Missing imageBase64');

    const buffer = Buffer.from(imageBase64, 'base64');
    const img = new Image();
    img.src = buffer;

    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const net = await bodyPix.load({ architecture: 'MobileNetV1', outputStride: 16 });
    const segmentation = await net.segmentPerson(canvas);

    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const newImageData = ctx.createImageData(imageData);

    for (let i = 0; i < segmentation.data.length; i++) {
      const isPerson = segmentation.data[i] === 1;
      newImageData.data[i * 4 + 0] = imageData.data[i * 4 + 0];
      newImageData.data[i * 4 + 1] = imageData.data[i * 4 + 1];
      newImageData.data[i * 4 + 2] = imageData.data[i * 4 + 2];
      newImageData.data[i * 4 + 3] = isPerson ? 255 : 0;
    }

    ctx.putImageData(newImageData, 0, 0);
    const result = canvas.toBuffer('image/png');
    const resultBase64 = result.toString('base64');

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error(e);
    res.status(500).send('Failed to remove background');
  }
}