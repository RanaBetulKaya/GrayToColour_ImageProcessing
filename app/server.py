from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import torch
from torchvision import transforms
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            #nn.Conv2d(in_size, out_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # print("Shape of x:", x.shape)
        # print("Shape of skip_input:", skip_input.shape)

        # x = F.interpolate(x, size=skip_input.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x, skip_input), 1)
        return x
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = Encoder(in_channels, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)
        self.down5 = Encoder(512, 512)
        self.down6 = Encoder(512, 512)
        self.down7 = Encoder(512, 512)
        self.down8 = Encoder(512, 512)

        self.up1 = Decoder(512, 512)
        self.up2 = Decoder(1024, 512)
        self.up3 = Decoder(1024, 512)
        self.up4 = Decoder(1024, 512)
        self.up5 = Decoder(1024, 256)
        self.up6 = Decoder(512, 128)
        self.up7 = Decoder(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.final(u7)

        return u8


# PyTorch modelini yükleme (eğer model .pth dosyasında kaydedildiyse)
#generator_model = torch.load("app/generator_model.pth", map_location=torch.device('cpu'))
#generator_model.eval()  # Modeli inference moduna alıyoruz

generator_model = GeneratorUNet()  # Model mimarisini yeniden tanımlıyoruz
generator_model.load_state_dict(torch.load("app/generator_model.pth", map_location=torch.device('cpu')))
generator_model.eval()

app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.get('/')
def read_route():
   return {'message': 'Model API'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        # Görüntüyü alıyoruz
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale olarak açıyoruz

        # Görüntüyü dönüştürüyoruz
        input_tensor = transform(image).unsqueeze(0)  # Batch dimension ekliyoruz
        
        # Modeli kullanarak tahmin yapıyoruz
        with torch.no_grad():
            output_tensor = generator_model(input_tensor)
        
        # Tahmin edilen görüntüyü yeniden boyutlandırıyoruz ve RGB formatına çeviriyoruz
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

        # Görüntüyü byte stream olarak kaydediyoruz
        byte_arr = io.BytesIO()
        output_image.save(byte_arr, format='PNG')  # PNG veya JPG olarak kaydedebilirsiniz
        byte_arr.seek(0)

        # Yanıt olarak PNG görüntüsünü döndürüyoruz
        return StreamingResponse(byte_arr, media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
