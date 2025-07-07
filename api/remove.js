import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import fetch from 'node-fetch';
import { join } from 'path';

// U2NET 模型的 URL
const MODEL_URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx';
let session = null;

async function loadModel() {
  if (session) return session;
  
  // 下載模型
  const response = await fetch(MODEL_URL);
  const modelBuffer = await response.arrayBuffer();
  
  // 創建 ONNX 會話
  session = await ort.InferenceSession.create(modelBuffer);
  return session;
}

async function preprocessImage(buffer) {
  // 調整圖片大小為 320x320
  const { data, info } = await sharp(buffer)
    .resize(320, 320, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .raw()
    .toBuffer({ resolveWithObject: true });

  // 正規化圖片數據
  const tensorData = new Float32Array(320 * 320 * 3);
  for (let i = 0; i < data.length; i += info.channels) {
    const offset = (i / info.channels) * 3;
    tensorData[offset] = data[i] / 255.0;     // R
    tensorData[offset + 1] = data[i + 1] / 255.0; // G
    tensorData[offset + 2] = data[i + 2] / 255.0; // B
  }

  return tensorData;
}

async function postprocessMask(mask, width, height) {
  // 將遮罩調整回原始大小
  return await sharp(Buffer.from(mask), {
    raw: {
      width: 320,
      height: 320,
      channels: 1
    }
  })
    .resize(width, height)
    .raw()
    .toBuffer();
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Only POST allowed');
  }

  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).send('Missing imageBase64');

    // 載入原始圖片
    const inputBuffer = Buffer.from(imageBase64, 'base64');
    const originalImage = sharp(inputBuffer);
    const { width, height } = await originalImage.metadata();

    // 載入模型
    const model = await loadModel();

    // 預處理圖片
    const preprocessedData = await preprocessImage(inputBuffer);

    // 創建輸入張量
    const tensor = new ort.Tensor('float32', preprocessedData, [1, 3, 320, 320]);

    // 運行推論
    const results = await model.run({ 'input.1': tensor });
    const output = results['output'];
    const maskData = output.data;

    // 後處理遮罩
    const processedMask = await postprocessMask(maskData, width, height);

    // 獲取原始圖片數據
    const { data: imageData } = await originalImage
      .raw()
      .toBuffer({ resolveWithObject: true });

    // 創建輸出緩衝區
    const outputData = new Uint8Array(width * height * 4);

    // 合併原始圖片和遮罩
    for (let i = 0; i < processedMask.length; i++) {
      const maskValue = processedMask[i];
      const inputIdx = i * 3;
      const outputIdx = i * 4;

      // 使用遮罩值作為透明度
      const alpha = maskValue > 128 ? 255 : 0;

      if (alpha > 0) {
        outputData[outputIdx] = imageData[inputIdx];     // R
        outputData[outputIdx + 1] = imageData[inputIdx + 1]; // G
        outputData[outputIdx + 2] = imageData[inputIdx + 2]; // B
        outputData[outputIdx + 3] = alpha;                   // A
      } else {
        outputData[outputIdx] = 0;
        outputData[outputIdx + 1] = 0;
        outputData[outputIdx + 2] = 0;
        outputData[outputIdx + 3] = 0;
      }
    }

    // 最終處理
    const finalBuffer = await sharp(outputData, {
      raw: {
        width,
        height,
        channels: 4
      }
    })
      .png()
      .toBuffer();

    // 轉換為 base64
    const resultBase64 = finalBuffer.toString('base64');

    res.status(200).json({ imageBase64: resultBase64 });
  } catch (e) {
    console.error('Error:', e.stack);
    res.status(500).send(`Failed to process image: ${e.message}`);
  }
}