/*
 * LD7182 — AI-Based Smart Environment Monitoring & Control System
 * ESP32-S3 + DHT22 + MQ-135 + LDR + PIR + LCD + Relay + Blynk
 *
 * On-device TinyML inference using a Random Forest classifier
 * exported from scikit-learn via micromlgen.
 *
 * Predicted classes:  0 = Normal   1 = Warning   2 = Critical
 */

#include <WiFi.h>
#define BLYNK_TEMPLATE_ID   "TMPL_XXXXXXXX"
#define BLYNK_TEMPLATE_NAME "SmartEnvAI"
#define BLYNK_AUTH_TOKEN    "YOUR_BLYNK_TOKEN"
#include <BlynkSimpleEsp32.h>
#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "RandomForest.h"   // micromlgen-generated header

// ---------- Pin map ----------
#define DHTPIN     4
#define DHTTYPE    DHT22
#define MQ_PIN     34
#define LDR_PIN    35
#define PIR_PIN    5
#define RELAY_PIN  25
#define BUZZER_PIN 26
#define LED_R      18
#define LED_G      19
#define LED_B      23

// ---------- Globals ----------
DHT dht(DHTPIN, DHTTYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);
Eloquent::ML::Port::RandomForest rf;

char ssid[] = "YOUR_WIFI";
char pass[] = "YOUR_PASS";

// ---------- Helpers ----------
void setStatusLED(int s) {
  digitalWrite(LED_R, s == 2);   // Critical = red
  digitalWrite(LED_G, s == 0);   // Normal   = green
  digitalWrite(LED_B, s == 1);   // Warning  = blue
}

void showStatusOnLCD(int s, float t, float h, float co2) {
  lcd.setCursor(0, 0);
  lcd.print("T:");  lcd.print(t, 1);
  lcd.print(" H:"); lcd.print(h, 0); lcd.print("%");

  lcd.setCursor(0, 1);
  if      (s == 0) lcd.print("NORMAL   ");
  else if (s == 1) lcd.print("WARNING  ");
  else             lcd.print("CRITICAL ");
  lcd.print("CO2:"); lcd.print((int)co2);
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(PIR_PIN,    INPUT);
  pinMode(RELAY_PIN,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_R, OUTPUT);
  pinMode(LED_G, OUTPUT);
  pinMode(LED_B, OUTPUT);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("SmartEnv AIoT");
  lcd.setCursor(0, 1); lcd.print("Booting...");

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  Serial.println("Boot complete.");
}

// ---------- Main loop ----------
void loop() {
  // 1. Read sensors
  float t   = dht.readTemperature();
  float h   = dht.readHumidity();
  float mq  = analogRead(MQ_PIN);
  float lux = analogRead(LDR_PIN);
  int   occ = digitalRead(PIR_PIN);

  // 2. Calibrate analogue gas sensor to engineering units
  float co2 = 350.0 + (mq / 4095.0) * 100.0;
  float aq  = 40.0  + (mq / 4095.0) * 20.0;

  // 3. Build feature vector and run on-device inference
  float features[6] = { t, h, co2, aq, lux, (float)occ };
  int   status      = rf.predict(features);

  // 4. Drive local actuators
  setStatusLED(status);
  digitalWrite(RELAY_PIN, status >= 1 ? HIGH : LOW);
  if (status == 2) tone(BUZZER_PIN, 2000, 200);

  // 5. Update LCD
  showStatusOnLCD(status, t, h, co2);

  // 6. Stream to Blynk cloud
  Blynk.virtualWrite(V0, t);
  Blynk.virtualWrite(V1, h);
  Blynk.virtualWrite(V2, co2);
  Blynk.virtualWrite(V3, aq);
  Blynk.virtualWrite(V4, occ);
  Blynk.virtualWrite(V5, status);

  Blynk.run();
  delay(15000);
}
