/*
 * Smart Wheelchair Safety Firmware for STM32F411
 *
 * Modular Architecture:
 * - Perception Layer: Ultrasonic (Left/Right) & LiDAR (Stub).
 * - Decision Layer: Logic outputting SafetyCommand (speed_scale, turn_bias, emergency_stop).
 * - Actuation Layer: Motor control, Kinematics, Ramping, Friction Deadband.
 *
 * Features: Non-blocking, fault tolerant, independent modules.
 */

#include <Arduino.h>
#include <math.h>

// ==========================================
// 1. HARDWARE PINS & CONSTANTS
// ==========================================
const uint8_t PIN_PWM_L = PA8;
const uint8_t PIN_DIR_L = PB0;
const uint8_t PIN_PWM_R = PA9;
const uint8_t PIN_DIR_R = PB1;

// Ultrasonic Pins (Independent L/R for this architecture)
// * User Must Verify/Wire the right-side correctly *
const uint8_t PIN_TRIG_L = PB10;
const uint8_t PIN_ECHO_L = PB4;
const uint8_t PIN_TRIG_R = PA4; // Right Trigger
const uint8_t PIN_ECHO_R = PB5;  // Right Echo

const uint32_t CMD_TIMEOUT_MS = 200;
const float WHEEL_BASE = 0.6;
const float V_MAX = 1.5;  // m/s
const float WZ_MAX = 1.5; // rad/s
const float PWM_MAX = 255.0;
const float MIN_PWM = 80.0; // Deadband compensation for 90kg load

const float MAX_ACCEL = 0.40;      // m/s^2
const float MAX_ANG_ACCEL = 1.0;   // rad/s^2
const float ACCEL_STEP = MAX_ACCEL / 100.0; // 100Hz loop step
const float ANG_STEP = MAX_ANG_ACCEL / 100.0;

// ==========================================
// 2. TIMING & INTERRUPTS
// ==========================================
volatile uint32_t echo_start_L = 0;
volatile uint32_t echo_dur_L = 0;
volatile bool new_echo_L = false;

volatile uint32_t echo_start_R = 0;
volatile uint32_t echo_dur_R = 0;
volatile bool new_echo_R = false;

void isr_echo_L() {
    if (digitalRead(PIN_ECHO_L)) {
        echo_start_L = micros();
    } else {
        echo_dur_L = micros() - echo_start_L;
        new_echo_L = true;
    }
}

void isr_echo_R() {
    if (digitalRead(PIN_ECHO_R)) {
        echo_start_R = micros();
    } else {
        echo_dur_R = micros() - echo_start_R;
        new_echo_R = true;
    }
}

// ==========================================
// 3. PERCEPTION LAYER
// ==========================================
class MedianFilter {
    float buf[3];
    uint8_t idx;
    bool filled;
public:
    MedianFilter() : idx(0), filled(false) {
        buf[0] = buf[1] = buf[2] = 999.0f;
    }
    void add(float val) {
        buf[idx++] = val;
        if (idx >= 3) { idx = 0; filled = true; }
    }
    float get() {
        if (!filled) return buf[0];
        float a = buf[0], b = buf[1], c = buf[2];
        if (a > b) { float t = a; a = b; b = t; }
        if (b > c) { float t = b; b = c; c = t; }
        if (a > b) { float t = a; a = b; b = t; }
        return b; // Return middle value
    }
};

class UltrasonicModule {
    enum State { IDLE, WAIT_L, WAIT_R };
    State state = IDLE;
    uint32_t state_time = 0;
    
    MedianFilter filter_L;
    MedianFilter filter_R;
    
    float dist_l = 999.0f;
    float dist_r = 999.0f;
    bool active = false;
    uint32_t last_valid_time = 0;
    
public:
    void init() {
        pinMode(PIN_TRIG_L, OUTPUT);
        pinMode(PIN_ECHO_L, INPUT);
        attachInterrupt(digitalPinToInterrupt(PIN_ECHO_L), isr_echo_L, CHANGE);
        
        pinMode(PIN_TRIG_R, OUTPUT);
        pinMode(PIN_ECHO_R, INPUT);
        attachInterrupt(digitalPinToInterrupt(PIN_ECHO_R), isr_echo_R, CHANGE);
    }
    
    void update() {
        uint32_t now = millis();
        
        // Fault-tolerance: Drop active flag if unresponsive for 1 second
        if (now - last_valid_time > 1000) active = false; 

        // Non-blocking Sequential State Machine
        switch (state) {
            case IDLE:
                trigger_left(now);
                break;
                
            case WAIT_L:
                if (new_echo_L) {
                    float d = (echo_dur_L * 0.0343f) / 2.0f;
                    filter_L.add(d);
                    dist_l = filter_L.get();
                    active = true;
                    last_valid_time = now;
                    trigger_right(now);
                } else if (now - state_time > 50) { // 50ms pulse timeout
                    trigger_right(now);
                }
                break;
                
            case WAIT_R:
                if (new_echo_R) {
                    float d = (echo_dur_R * 0.0343f) / 2.0f;
                    filter_R.add(d);
                    dist_r = filter_R.get();
                    active = true;
                    last_valid_time = now;
                    state = IDLE;
                } else if (now - state_time > 50) {
                    state = IDLE;
                }
                break;
        }
    }
    
    bool is_active() const { return active; }
    float get_distance_left() const { return dist_l; }
    float get_distance_right() const { return dist_r; }

private:
    void trigger_left(uint32_t now) {
        digitalWrite(PIN_TRIG_L, HIGH);
        delayMicroseconds(10);
        digitalWrite(PIN_TRIG_L, LOW);
        new_echo_L = false;
        state = WAIT_L;
        state_time = now;
    }
    
    void trigger_right(uint32_t now) {
        digitalWrite(PIN_TRIG_R, HIGH);
        delayMicroseconds(10);
        digitalWrite(PIN_TRIG_R, LOW);
        new_echo_R = false;
        state = WAIT_R;
        state_time = now;
    }
};

class LidarModule {
    bool active = false;
    float front_dist = 999.0f;
public:
    void init() {
        // Stub for future serial integration
    }
    void update() {
        // Architecture handles independent integration cleanly
    }
    bool is_active() const { return active; }
    float get_front_distance() const { return front_dist; }
};

class PerceptionLayer {
    UltrasonicModule us;
    LidarModule lidar;
public:
    void init() { us.init(); lidar.init(); }
    void update() { us.update(); lidar.update(); }
    
    // Abstracted Interface
    float dist_left() { return us.get_distance_left(); }
    float dist_right() { return us.get_distance_right(); }
    float dist_front() { return lidar.get_front_distance(); }
    bool us_active() { return us.is_active(); }
    bool lidar_active() { return lidar.is_active(); }
};

// ==========================================
// 4. DECISION LAYER
// ==========================================
struct SafetyCommand {
    float speed_scale; // 0.0 to 1.0 multiplier
    float turn_bias;   // -1.0 to 1.0 injection
    bool emergency_stop;
};

class DecisionLayer {
public:
    SafetyCommand evaluate(PerceptionLayer& perception) {
        SafetyCommand cmd = {1.0f, 0.0f, false};
        
        bool us_on = perception.us_active();
        bool li_on = perception.lidar_active();

        // BEHAVIORAL RULE: Manual Fallback If No Sensors Active
        if (!us_on && !li_on) {
            return cmd; // Unaltered input allowed
        }
        
        // ULTRASONIC RULES
        if (us_on) {
            float dl = perception.dist_left();
            float dr = perception.dist_right();
            
            // Critical Event
            if (dl < 30.0f && dr < 30.0f) {
                cmd.emergency_stop = true;
                return cmd;
            }
            
            // Proximity Constraint
            if (dl < 60.0f || dr < 60.0f) {
                float min_dist = min(dl, dr);
                if (min_dist < 30.0f) min_dist = 30.0f;
                // Scale ranges 0.0 to 1.0 proportionally constraint
                cmd.speed_scale = (min_dist - 30.0f) / 30.0f;
                
                // Obstacle Avoidance Turn Bias
                if (abs(dl - dr) > 10.0f) { 
                    if (dl < dr) cmd.turn_bias = -0.5f; // Bias to Right
                    else cmd.turn_bias = 0.5f;          // Bias to Left
                }
            }
        }
        
        // LiDAR Rules (Future)
        // if (li_on) { ... apply further constraints }
        
        return cmd;
    }
};

// ==========================================
// 5. ACTUATION LAYER
// ==========================================
class ActuationLayer {
    float current_vx = 0.0f;
    float current_wz = 0.0f;
    uint32_t last_control_time = 0;

public:
    void init() {
        pinMode(PIN_PWM_L, OUTPUT);
        pinMode(PIN_DIR_L, OUTPUT);
        pinMode(PIN_PWM_R, OUTPUT);
        pinMode(PIN_DIR_R, OUTPUT);
    }

    void update(float user_vx, float user_wz, SafetyCommand safety) {
        uint32_t now = millis();
        if (now - last_control_time < 10) return; // Maintain strict 100Hz control loop
        last_control_time = now;

        float target_vx = 0.0f;
        float target_wz = 0.0f;

        // Constraint Integration
        if (safety.emergency_stop) {
            target_vx = 0.0f;
            target_wz = 0.0f;
            current_vx = 0.0f; // Fast Hardware Hardware Stop
            current_wz = 0.0f;
        } else {
            target_vx = user_vx * safety.speed_scale;
            target_wz = user_wz;
            
            if (safety.turn_bias != 0.0f) {
                target_wz += safety.turn_bias * WZ_MAX; 
                target_wz = constrain(target_wz, -WZ_MAX, WZ_MAX);
            }

            // Smooth Ramping Kinematics
            if (current_vx < target_vx) {
                current_vx += ACCEL_STEP;
                if (current_vx > target_vx) current_vx = target_vx;
            } else if (current_vx > target_vx) {
                current_vx -= ACCEL_STEP;
                if (current_vx < target_vx) current_vx = target_vx;
            }

            if (current_wz < target_wz) {
                current_wz += ANG_STEP;
                if (current_wz > target_wz) current_wz = target_wz;
            } else if (current_wz > target_wz) {
                current_wz -= ANG_STEP;
                if (current_wz < target_wz) current_wz = target_wz;
            }
        }

        // Differential Drive Math
        float v_left = current_vx - (current_wz * WHEEL_BASE / 2.0f);
        float v_right = current_vx + (current_wz * WHEEL_BASE / 2.0f);

        // Deadband Comp & PWM Scaling
        int32_t pwm_l = 0;
        if (abs(v_left) > 0.02f) {
            pwm_l = (int32_t)((abs(v_left) / V_MAX) * (PWM_MAX - MIN_PWM) + MIN_PWM);
            if (v_left < 0) pwm_l = -pwm_l;
        }

        int32_t pwm_r = 0;
        if (abs(v_right) > 0.02f) {
            pwm_r = (int32_t)((abs(v_right) / V_MAX) * (PWM_MAX - MIN_PWM) + MIN_PWM);
            if (v_right < 0) pwm_r = -pwm_r;
        }

        pwm_l = constrain(pwm_l, -PWM_MAX, PWM_MAX);
        pwm_r = constrain(pwm_r, -PWM_MAX, PWM_MAX);

        digitalWrite(PIN_DIR_L, pwm_l >= 0 ? HIGH : LOW); 
        digitalWrite(PIN_DIR_R, pwm_r >= 0 ? LOW : HIGH);
        analogWrite(PIN_PWM_L, abs(pwm_l));
        analogWrite(PIN_PWM_R, abs(pwm_r));
    }
};

// ==========================================
// 6. MAIN SYSTEM INTEGRATION
// ==========================================
PerceptionLayer perception;
DecisionLayer decision;
ActuationLayer actuation;

float user_vx = 0.0f;
float user_wz = 0.0f;
uint32_t last_cmd_time = 0;
char cmd_buf[128];

void process_serial_command(const char* s) {
    long vx_mm = 0;
    long wz_mrad = 0;

    if (sscanf(s, "M,%ld,%ld", &vx_mm, &wz_mrad) == 2) {
        float vx = vx_mm / 1000.0f;
        float wz = wz_mrad / 1000.0f;

        if (isfinite(vx) && isfinite(wz)) {
            user_vx = constrain(vx, 0.0f, V_MAX); // Forward only safety enforced here
            user_wz = constrain(wz, -WZ_MAX, WZ_MAX);
            last_cmd_time = millis();
        }
    }
}

void setup() {
    SerialUSB.begin(115200);
    perception.init();
    actuation.init();
    last_cmd_time = millis();
}

void loop() {
    uint32_t now = millis();

    // 1. Read Serial Command Input (Non-blocking)
    while (SerialUSB.available()) {
        char c = SerialUSB.read();
        if (c == '\n' || c == '\r') {
            if (strlen(cmd_buf) > 0) {
                process_serial_command(cmd_buf);
                cmd_buf[0] = '\0';
            }
        } else {
            size_t len = strlen(cmd_buf);
            if (len < sizeof(cmd_buf)-1) {
                cmd_buf[len] = c;
                cmd_buf[len+1] = '\0';
            }
        }
    }

    // 2. Dead-mans Switch Fallback (Network / Process Loss)
    if (now - last_cmd_time > CMD_TIMEOUT_MS) {
        user_vx = 0.0f;
        user_wz = 0.0f;
    }

    // 3. Perception Polling
    perception.update();

    // 4. Decision Processing
    SafetyCommand safe_cmd = decision.evaluate(perception);

    // 5. Motor Actuation
    actuation.update(user_vx, user_wz, safe_cmd);
    
    // 6. Telemetry Reporting (10Hz)
    static uint32_t last_telem_time = 0;
    if (now - last_telem_time > 100) {
        last_telem_time = now;
        SerialUSB.print("US;");
        if (perception.us_active()) {
            SerialUSB.print("L:"); SerialUSB.print((int)perception.dist_left()); SerialUSB.print(";");
            SerialUSB.print("R:"); SerialUSB.print((int)perception.dist_right());
        } else {
            SerialUSB.print("OFF");
        }
        SerialUSB.print("\n");
    }
}
