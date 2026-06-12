/**
 * ╔════════════════════════════════════════════════════════════════════════════════╗
 * ║                  OCULAR INTENT WHEELCHAIR — ESP8266 FIRMWARE v3.2              ║
 * ║           Serial Bridge: Blynk Interface ↔ Raspberry Pi ROS2 Agent             ║
 * ╚════════════════════════════════════════════════════════════════════════════════╝
 *
 * PLATFORMIO BUILD: Exact syntax matching for the 'wollewald/INA226_WE' library.
 * → Read Call Swapped: Changed to .getBusVoltage_V()
 * → Setup Configuration: Updated enums to INA226_AVERAGE_128 and .setConversionTime()
 * → Range: 23.18V (0%) to 25.29V (100%) Lead-Acid Profile.
 */

#include <Wire.h>
#include <INA226_WE.h>
#include <time.h>
#include <ArduinoJson.h>

// === SYSTEM CREDENTIALS ===
#define BLYNK_TEMPLATE_ID   "TMPL2PTA3PrSL"
#define BLYNK_TEMPLATE_NAME "Wheelchair Interface"
#define BLYNK_AUTH_TOKEN    "l_tor_0Bv9N4_PDUNDzw0FedZQZD3zmP"

#define FIRMWARE_VERSION "3.2-8266"

// === MULTIPLE WiFi NETWORKS ===
struct WiFiNetwork {
    const char* ssid;
    const char* password;
};

const int WIFI_NETWORKS_COUNT = 3;
WiFiNetwork wifiNetworks[WIFI_NETWORKS_COUNT] = {
    {"Mohamed", "Mohamed*123"},      // Primary network
    {"192.168.u.0", "mo123456"},  // Backup network 1
    {"Guest_Network", "guest_pass"}  // Backup network 2
};

int currentNetworkIndex = 0;

// === NETWORK & BLYNK INCLUDES ===
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>

// === HARDWARE INSTANCES ===
#define I2C_ADDRESS 0x40
INA226_WE ina = INA226_WE(I2C_ADDRESS); 
BlynkTimer timer;

// === SYSTEM STATE VARIABLES ===
int currentMode = 0;                    
int joyX = 0;                            
int joyY = 0;                            
bool hardwarePresent = false;           

// === BUTTON STATE REGISTERS ===
int btnForward = 0;
int btnReverse = 0;
int btnLeft    = 0;
int btnRight   = 0;

// === SAFETY TIMEOUT TRACKING ===
unsigned long companionStartTime = 0;
const unsigned long COMPANION_TIMEOUT_MS = 300000;  
unsigned long lastTimeoutLogTime = 0;                

// === WiFi RECONNECTION & LED BLINK TRACKING ===
unsigned long lastWiFiAttempt = 0;
const unsigned long WIFI_RETRY_INTERVAL = 10000;   
unsigned long lastLEDBlinkTime = 0;
const unsigned long LED_BLINK_INTERVAL_MS = 200; 
bool ledState = false;

// === BATTERY SOC LOOKUP TABLE ===
struct VoltageMapping {
    float voltage;
    float percentage;
};

// Calibrated 24V Flooded Lead-Acid Table
const int TABLE_SIZE = 11;
VoltageMapping socTable[TABLE_SIZE] = {
    {25.29, 100.0}, 
    {25.05, 90.0}, 
    {24.81, 80.0}, 
    {24.58, 70.0},
    {24.36, 60.0},  
    {24.14, 50.0}, 
    {23.94, 40.0}, 
    {23.74, 30.0},
    {23.51, 20.0},  
    {23.27, 10.0}, 
    {23.18, 0.0}
};

float calculatePercentage(float voltage) {
    if (voltage <= 0.05) return 0.0;
    
    if (voltage >= socTable[0].voltage) return 100.0;
    if (voltage <= socTable[TABLE_SIZE - 1].voltage) return 0.0;
    
    for (int i = 0; i < TABLE_SIZE - 1; i++) {
        if (voltage <= socTable[i].voltage && voltage >= socTable[i+1].voltage) {
            return socTable[i].percentage + 
                   ((voltage - socTable[i].voltage) / (socTable[i+1].voltage - socTable[i].voltage)) * (socTable[i+1].percentage - socTable[i].percentage);
        }
    }
    return 0.0; 
}

float getSystemVoltage() {
    if (hardwarePresent) {
        // Updated to the exact method suggested by the compiler
        float readVolt = ina.getBusVoltage_V();
        
        if (readVolt <= 0.05) {
            return 0.0;
        }
        return readVolt;
    } 
    return 0.0; 
}

void sendSystemStateToPi(float currentVoltage) {
    float batteryPercentage = calculatePercentage(currentVoltage);
    int linkStatus = (WiFi.status() == WL_CONNECTED && Blynk.connected()) ? 1 : 0;
    uint32_t timestamp = millis() / 1000;

    // --- DIAGNOSTIC HUMAN TERMINAL LOGGING ---
    Serial.println("\n--- [DIAGNOSTIC TRACE] ---");
    Serial.print("  I2C WE Driver Status : "); Serial.println(hardwarePresent ? "ONLINE (0x40)" : "OFFLINE/FAILED");
    Serial.print("  Measured Bus Voltage : "); Serial.print(currentVoltage, 3); Serial.println(" V");
    Serial.print("  Computed Percent SOC : "); Serial.print(batteryPercentage, 1); Serial.println(" %");
    Serial.print("  Active System Mode   : "); Serial.println(currentMode == 1 ? "Companion (Blynk)" : "Patient (Eye-Tracking)");
    Serial.print("  Target Coordinates   : X="); Serial.print(joyX); Serial.print(", Y="); Serial.println(joyY);
    Serial.println("--------------------------");

    // --- LOW-OVERHEAD JSON STREAM FOR THE PI ---
    Serial.print("{\"mode\":");
    Serial.print(currentMode);
    Serial.print(",\"x\":");
    Serial.print(joyX);
    Serial.print(",\"y\":");
    Serial.print(joyY);
    Serial.print(",\"volt\":");       
    Serial.print(currentVoltage, 2);
    Serial.print(",\"bat\":");
    Serial.print(batteryPercentage, 1);
    Serial.print(",\"hw_ok\":");
    Serial.print(hardwarePresent ? 1 : 0);
    Serial.print(",\"link\":");
    Serial.print(linkStatus);
    Serial.print(",\"ts\":");
    Serial.print(timestamp);
    Serial.print(",\"fw\":\"");
    Serial.print(FIRMWARE_VERSION);
    Serial.println("\"}");
}

void clearMotionVectors() {
    joyX = 0;
    joyY = 0;
    btnForward = 0;
    btnReverse = 0;
    btnLeft    = 0;
    btnRight   = 0;
}

void evaluateMotionMatrices() {
    if (btnForward && !btnReverse)       joyY = 255;
    else if (btnReverse && !btnForward)  joyY = -255;
    else                                 joyY = 0; 

    if (btnRight && !btnLeft)            joyX = 255;
    else if (btnLeft && !btnRight)       joyX = -255;
    else                                 joyX = 0; 
}

void processSerialInput() {
    if (!Serial.available()) return;
    
    int incomingByte = Serial.read();
    if (incomingByte != '>') return;
    
    String commandLine = "";
    int timeout = 0;
    while (Serial.available() && timeout < 50) {
        int c = Serial.read();
        if (c == '\n') break;
        commandLine += (char)c;
        timeout++;
    }
    
    if (commandLine.length() == 0) return;
    
    StaticJsonDocument<128> doc;
    DeserializationError error = deserializeJson(doc, commandLine);
    if (error) return;
    
    const char* cmd = doc["cmd"];
    if (!cmd) return;
    
    if (strcmp(cmd, "req_state") == 0) {
        float currentVoltage = getSystemVoltage();
        sendSystemStateToPi(currentVoltage);
    }
    else if (strcmp(cmd, "set_mode") == 0) {
        int newMode = doc["val"];
        if (newMode != 0 && newMode != 1) return;
        currentMode = newMode;
        
        clearMotionVectors();
        if (currentMode == 1) {
            companionStartTime = millis();
            lastTimeoutLogTime = millis();
        }
        
        if (Blynk.connected()) {
            Blynk.virtualWrite(V3, currentMode);
            Blynk.virtualWrite(V4, 0);
            Blynk.virtualWrite(V5, 0);
            Blynk.virtualWrite(V6, 0);
            Blynk.virtualWrite(V7, 0);
        }
        float currentVoltage = getSystemVoltage();
        sendSystemStateToPi(currentVoltage);
    }
}

BLYNK_WRITE(V3) {
    currentMode = param.asInt();
    clearMotionVectors();
    if (currentMode == 1) {
        companionStartTime = millis();
        lastTimeoutLogTime = millis();
    }
    
    Blynk.virtualWrite(V4, 0);
    Blynk.virtualWrite(V5, 0);
    Blynk.virtualWrite(V6, 0);
    Blynk.virtualWrite(V7, 0);

    float currentVoltage = getSystemVoltage();
    sendSystemStateToPi(currentVoltage);
}

BLYNK_WRITE(V4) { btnForward = param.asInt(); if (currentMode == 1) { evaluateMotionMatrices(); sendSystemStateToPi(getSystemVoltage()); } }
BLYNK_WRITE(V5) { btnReverse = param.asInt(); if (currentMode == 1) { evaluateMotionMatrices(); sendSystemStateToPi(getSystemVoltage()); } }
BLYNK_WRITE(V6) { btnLeft    = param.asInt(); if (currentMode == 1) { evaluateMotionMatrices(); sendSystemStateToPi(getSystemVoltage()); } }
BLYNK_WRITE(V7) { btnRight   = param.asInt(); if (currentMode == 1) { evaluateMotionMatrices(); sendSystemStateToPi(getSystemVoltage()); } }

void systemHeartbeat() {
    float currentVoltage = getSystemVoltage();
    float batteryPercentage = calculatePercentage(currentVoltage);

    if (WiFi.status() == WL_CONNECTED && Blynk.connected()) {
        Blynk.virtualWrite(V1, batteryPercentage);
    }
    sendSystemStateToPi(currentVoltage);
}

void handleNetworkLED(bool connected) {
    if (connected) {
        digitalWrite(LED_BUILTIN, LOW); 
    } else {
        if (millis() - lastLEDBlinkTime >= LED_BLINK_INTERVAL_MS) {
            ledState = !ledState;
            digitalWrite(LED_BUILTIN, ledState ? LOW : HIGH);
            lastLEDBlinkTime = millis();
        }
    }
}

bool attemptWiFiConnection(int networkIndex) {
    if (networkIndex >= WIFI_NETWORKS_COUNT) return false;
    
    const char* ssid = wifiNetworks[networkIndex].ssid;
    const char* pass = wifiNetworks[networkIndex].password;
    
    Serial.print("[WiFi] Attempting connection to: ");
    Serial.println(ssid);
    
    WiFi.begin(ssid, pass);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        currentNetworkIndex = networkIndex;
        Serial.print("[WiFi] Connected to: ");
        Serial.println(ssid);
        return true;
    }
    
    return false;
}

void tryNextWiFiNetwork() {
    int startIndex = currentNetworkIndex;
    
    for (int i = 0; i < WIFI_NETWORKS_COUNT; i++) {
        int nextIndex = (startIndex + i + 1) % WIFI_NETWORKS_COUNT;
        if (attemptWiFiConnection(nextIndex)) {
            return;
        }
    }
    
    Serial.println("[WiFi] Failed to connect to any available network");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH); 
    
    Wire.begin(4, 5); // SDA=GPIO4, SCL=GPIO5
    
    // Explicit WE Library Initialization sequence
    if (ina.init()) {
        hardwarePresent = true;
        
        // Synced configuration routines with the exact API library maps
        ina.setAverage(INA226_AVERAGE_128);
        ina.setConversionTime(INA226_CONV_TIME_1100);
        ina.setMeasureMode(INA226_CONTINUOUS);
        
        if (ina.getBusVoltage_V() <= 0.05) {
            Serial.println("[WARN] INA226 initialized but VBUS reads 0.0V. Verify your wiring!");
        }
    } else {
        hardwarePresent = false;
        Serial.println("[CRITICAL] INA226 hardware failed handshake over Wire pins 4 and 5!");
    }
    
    WiFi.mode(WIFI_STA);
    attemptWiFiConnection(0); // Try first network from list
    
    Blynk.config(BLYNK_AUTH_TOKEN);
    timer.setInterval(2000L, systemHeartbeat);
    
    ESP.wdtEnable(WDTO_8S); 
}

void loop() {
    ESP.wdtFeed();
    processSerialInput();
    
    bool networkIsAlive = (WiFi.status() == WL_CONNECTED);
    handleNetworkLED(networkIsAlive);
    
    if (!networkIsAlive) {
        if (millis() - lastWiFiAttempt >= WIFI_RETRY_INTERVAL) {
            tryNextWiFiNetwork();
            lastWiFiAttempt = millis();
        }
        
        if (currentMode != 0) {
            currentMode = 0; clearMotionVectors();
            sendSystemStateToPi(getSystemVoltage());
        }
    } else {
        Blynk.run();
        
        if (!Blynk.connected() && currentMode != 0) {
            currentMode = 0; clearMotionVectors();
            sendSystemStateToPi(getSystemVoltage());
        }
    }

    if (currentMode == 1) {
        unsigned long elapsed = millis() - companionStartTime;
        if (elapsed >= COMPANION_TIMEOUT_MS) {
            currentMode = 0; clearMotionVectors();
            if (Blynk.connected()) {
                Blynk.virtualWrite(V3, 0);
                Blynk.virtualWrite(V4, 0);
                Blynk.virtualWrite(V5, 0);
                Blynk.virtualWrite(V6, 0);
                Blynk.virtualWrite(V7, 0);
            }
            sendSystemStateToPi(getSystemVoltage());
        }
    }

    timer.run();
    yield();
}