import { removeBackground } from '@rembg/node';
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

    // 使用 rembg 移除背景
    const outputBuffer = await removeBackground(preprocessedBuffer, {
      model: 'u2net_cloth', // 使用專門的服飾模型
      postprocessing: {
        alpha_matting: true,
        alpha_matting_foreground_threshold: 240,
        alpha_matting_background_threshold: 10,
        alpha_matting_erode_size: 10
      }
    });

    // 後處理：優化邊緣和透明度
    const finalBuffer = await sharp(outputBuffer)
      // 輕微模糊以平滑邊緣
      .blur(0.3)
      // 調整透明度閾值
      .bandbool(3, 'and')
      // 確保輸出為 PNG 以保持透明度
      .png({
        quality: 100,
        compressionLevel: 9
      })
      .toBuffer();

    // 轉換回 base64
    const resultBase64 = finalBuffer.toString('base64');

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error('Error:', e.stack);
    res.status(500).send(`Failed to process image: ${e.message}`);
  }
}