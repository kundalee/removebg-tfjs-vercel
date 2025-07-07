import { removeBackground } from '@imgly/background-removal';
import sharp from 'sharp';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST allowed');
  }

  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).send('Missing imageBase64');

    // 將 base64 轉換為 buffer
    const inputBuffer = Buffer.from(imageBase64, 'base64');

    // 使用 sharp 進行預處理
    const preprocessedBuffer = await sharp(inputBuffer)
      // 確保圖片不會太大，但保持足夠的細節
      .resize(1024, 1024, {
        fit: 'inside',
        withoutEnlargement: true
      })
      // 增強對比度以幫助分割
      .modulate({
        brightness: 1.05,
        contrast: 1.1
      })
      .toBuffer();

    // 使用 imgly 移除背景
    const outputBuffer = await removeBackground(preprocessedBuffer, {
      quality: 'best',
      refine: true,
      format: 'png'
    });

    // 後處理：優化輸出
    const finalBuffer = await sharp(outputBuffer)
      // 輕微銳化以保持細節
      .sharpen(0.5)
      // 確保輸出為高品質 PNG
      .png({
        quality: 100,
        compressionLevel: 9
      })
      .toBuffer();

    // 轉換為 base64
    const resultBase64 = finalBuffer.toString('base64');

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error('Error:', e.stack);
    res.status(500).send(`Failed to process image: ${e.message}`);
  }
}