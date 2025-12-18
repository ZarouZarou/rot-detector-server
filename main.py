from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import tempfile
import librosa
import traceback
import os

app = FastAPI()

# Charger le modèle
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Modèle chargé avec succès")
    print("🔍 Input shape attendue:", input_details[0]['shape'])
    print("🔍 Output shape:", output_details[0]['shape'])
except Exception as e:
    print(f"❌ ERREUR CRITIQUE : Impossible de charger le modèle : {e}")
    exit(1)

SEGMENT_LENGTH = 44032
ROT_THRESHOLD = 0.80
MAX_AUDIO_DURATION = 30

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    tmp_path = None
    
    try:
        # Vérifier la taille du fichier (max 10MB)
        content = await audio.read()
        if len(content) > 10 * 1024 * 1024:
            print(f"❌ Fichier trop gros : {len(content)} bytes")
            return {"is_rot": False, "confidence": 0.0, "error": "File too large (max 10MB)"}
        
        if len(content) == 0:
            print("❌ Fichier vide reçu")
            return {"is_rot": False, "confidence": 0.0, "error": "Empty audio file"}
        
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"📥 Fichier reçu : {len(content)} bytes")
        
        # Charger l'audio
        try:
            data, sr = librosa.load(tmp_path, sr=44100, mono=True, duration=MAX_AUDIO_DURATION)
        except Exception as e:
            print(f"❌ Erreur lors du chargement audio : {e}")
            return {"is_rot": False, "confidence": 0.0, "error": "Invalid audio format"}
        
        audio_duration = len(data) / sr
        print(f"📊 Audio : {len(data)} samples ({audio_duration:.2f}s) à {sr} Hz")
        
        # Vérifier durée minimale
        if audio_duration < 0.1:
            print("❌ Audio trop court")
            return {"is_rot": False, "confidence": 0.0, "error": "Audio too short"}
        
        # Vérifier que l'audio n'est pas juste du silence
        if np.abs(data).max() < 0.001:
            print("❌ Audio vide ou silence total")
            return {"is_rot": False, "confidence": 0.0, "error": "Silent audio"}
        
        # Découper en segments
        num_segments = int(np.ceil(len(data) / SEGMENT_LENGTH))
        print(f"🔪 Découpage en {num_segments} segment(s)")
        
        max_rot_confidence = 0.0
        rot_detected = False
        valid_segments = 0
        
        for i in range(num_segments):
            start = i * SEGMENT_LENGTH
            end = start + SEGMENT_LENGTH
            segment = data[start:end]
            
            # Si le segment est trop court, le répéter
            if len(segment) < SEGMENT_LENGTH:
                repeats = int(np.ceil(SEGMENT_LENGTH / len(segment)))
                segment = np.tile(segment, repeats)[:SEGMENT_LENGTH]
            
            # Préparer l'input
            input_data = np.expand_dims(segment, axis=0).astype(np.float32)
            
            # Inférence
            try:
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            except Exception as e:
                print(f"⚠️ Segment {i+1}: Erreur d'inférence : {e}")
                continue
            
            # Vérifier NaN/Inf
            if np.any(np.isnan(output_data)) or np.any(np.isinf(output_data)):
                print(f"⚠️ Segment {i+1}: Valeurs invalides, ignoré")
                continue
            
            valid_segments += 1
            
            # Appliquer softmax si nécessaire
            if output_data.max() > 1.0 or output_data.min() < 0.0:
                exp_values = np.exp(output_data - np.max(output_data))
                output_data = exp_values / exp_values.sum()
            
            # Classes: [Background Noise, Rot]
            rot_confidence = float(output_data[1])
            max_rot_confidence = max(max_rot_confidence, rot_confidence)
            
            print(f"  Segment {i+1}/{num_segments} ({start/sr:.2f}-{end/sr:.2f}s): "
                  f"Background={output_data[0]:.2%}, Rot={rot_confidence:.2%}")
            
            # Détection de rot
            if rot_confidence >= ROT_THRESHOLD:
                rot_detected = True
                print(f"  🎉 ROT DÉTECTÉ dans segment {i+1} !")
        
        # Si aucun segment valide, erreur
        if valid_segments == 0:
            print("❌ Aucun segment valide analysé")
            return {"is_rot": False, "confidence": 0.0, "error": "No valid segments"}
        
        print(f"\n{'='*50}")
        print(f"📈 Confiance Rot max: {max_rot_confidence:.2%}")
        print(f"🎯 Résultat final: {'ROT' if rot_detected else 'PAS DE ROT'}")
        print(f"{'='*50}\n")
        
        return {
            "is_rot": rot_detected,
            "confidence": round(float(max_rot_confidence), 4)
        }
    
    except Exception as e:
        print("❌ ERREUR:")
        traceback.print_exc()
        return {"is_rot": False, "confidence": 0.0, "error": str(e)}
    
    finally:
        # Toujours nettoyer le fichier temporaire
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"🗑️ Fichier temporaire supprimé")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer le fichier : {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage du serveur...")
    uvicorn.run(app, host="0.0.0.0", port=8000)