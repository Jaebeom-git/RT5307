#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <SPI.h>
#include <SD.h>
#define OFF 0
#define ON 1

// 센서 인스턴스
Adafruit_BNO055 bno1 = Adafruit_BNO055(0, 0x28, &Wire);
Adafruit_BNO055 bno2 = Adafruit_BNO055(1, 0x28, &Wire1);

// SD 카드 설정
const int cs = 52; // SD 카드 CS (Chip Select) 핀
File myFile;

// 데이터 저장 횟수
int data_num = 0;
String fileName;
int once = OFF;

// 데이터 개수
int total = 0;
float currentTime;

// SW/LED 제어
int sw1 = 23;
int led = 25;
int state = OFF;  // LED의 현재 상태를 저장하는 변수
int lastButtonState = HIGH;  // 마지막 버튼 상태 저장, 스위치 풀업 저항 사용으로 초기값 HIGH
int initialTime;
bool flag_sr = true;

void dataAquisition();
void setup() {
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  // LED, SWITCH
  pinMode(led, OUTPUT);
  pinMode(sw1, INPUT_PULLUP);
  pinMode(53, OUTPUT);

  // 센서 초기화
  Wire.begin();
  delay(500);
  Wire1.begin();
  delay(500);
  while (!bno1.begin() || !bno2.begin()) {
    delay(500);
  }

  bno1.setExtCrystalUse(true);
  bno2.setExtCrystalUse(true);

  // SD 카드 초기화
  delay(500);
  while (!SD.begin(cs)) {
    delay(500);
  }
  digitalWrite(led, HIGH);
  initialTime = int(millis());
}

void loop() {
  int buttonState = digitalRead(sw1);  // 현재 버튼 상태 읽기
  int deltaTime = int(millis());

  if (flag_sr) {
    digitalWrite(53, HIGH);
    flag_sr = false;
  } else {
    digitalWrite(53, LOW);
    flag_sr = true;
  }

  if (state == ON) {
    if (deltaTime % 1000 / 100 == 0) {
      digitalWrite(led, HIGH); // LED 켜기
    } else if (deltaTime % 1000 / 100 == 5) {
      digitalWrite(led, LOW); // LED 끄기
    }
  }

  // 버튼 상태가 변했는지, 그리고 버튼이 눌렸는지 확인
  if (buttonState != lastButtonState) {
    lastButtonState = buttonState;
    if (buttonState == LOW) { // 버튼이 눌렸다면
      if (state == OFF) {
        state = ON;
        delay(500);
        // SD 카드에 데이터 기록
        String fileName = String(data_num++) + ".csv"; // 새 파일 이름 생성
        myFile = SD.open(fileName.c_str(), FILE_WRITE); // 파일 열기
        if (myFile) {
          once = ON;
        }
      } else {
        state = OFF;
        digitalWrite(led, HIGH); // LED 끄기
        delay(500);
      }
    }
  }
  if (state == ON) dataAquisition();
  // 현재 버튼 상태를 마지막 버튼 상태로 저장
  lastButtonState = buttonState;
}

void dataAquisition() {
  // 헤더는 한 번만(once)
  if (once == ON) {
    String headers[] = {"foot", "shank"};
    myFile.print("Timestamp(s),");
    for (int i = 0; i < 2; i++) {
      myFile.print(headers[i] + "_Accel_X,");
      myFile.print(headers[i] + "_Accel_Y,");
      myFile.print(headers[i] + "_Accel_Z,");
      myFile.print(headers[i] + "_Gyro_X,");
      myFile.print(headers[i] + "_Gyro_Y,");
      myFile.print(headers[i] + "_Gyro_Z,");
    }
    myFile.println();
    once = OFF;
  }

  Adafruit_BNO055* sensors[2] = {&bno1, &bno2};
  for (int i = 0; i < 2; i++) {
    imu::Vector<3> accel = sensors[i]->getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    imu::Vector<3> gyro = sensors[i]->getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    if (i == 0) {
      currentTime = millis() / 1000.0;
      myFile.print(currentTime); myFile.print(","); // 타임스탬프 출력
    }
    myFile.print(accel.x()); myFile.print(",");
    myFile.print(accel.y()); myFile.print(",");
    myFile.print(accel.z()); myFile.print(",");
    myFile.print(gyro.x()); myFile.print(",");
    myFile.print(gyro.y()); myFile.print(",");
    myFile.print(gyro.z()); myFile.print(",");
  }
  myFile.println();
  total++;
  // 파일 닫기
  myFile.flush(); // Ensure data is written to the file
}
