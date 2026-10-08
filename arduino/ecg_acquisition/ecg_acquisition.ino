/*
  ECG Acquisition — AD8232 + Arduino UNO

  Reads the analog ECG waveform from the AD8232 module on pin A0 and
  streams it over USB serial as plain integer ADC values (0-1023),
  one per line, at a fixed sample rate. Also monitors the LO+/LO-
  (lead-off detection) pins and reports "LO" when an electrode is
  disconnected, so the host-side Python script can flag bad reads.

  Wiring:
    AD8232 OUTPUT -> Arduino A0
    AD8232 LO+    -> Arduino D10
    AD8232 LO-    -> Arduino D11
    AD8232 3.3V   -> Arduino 3.3V
    AD8232 GND    -> Arduino GND

  Electrodes (standard 3-lead placement):
    RA (right arm), LA (left arm), RL (right leg / ground reference)
*/

const int ECG_PIN = A0;
const int LO_PLUS = 10;
const int LO_MINUS = 11;

const unsigned long SAMPLE_RATE_HZ = 250;  // matches typical AD8232 use
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = now;

    if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
      Serial.println("LO");  // lead-off detected
    } else {
      int value = analogRead(ECG_PIN);
      Serial.println(value);
    }
  }
}
