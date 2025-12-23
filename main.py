from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import tempfile
import librosa
import traceback
import os
from supabase import create_client, Client
from typing import List

app = FastAPI()

# ✅ Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://rgbvspwzirfkdvbtvezg.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJnYnZzcHd6aXJma2R2YnR2ZXpnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYyNjQxMjYsImV4cCI6MjA4MTg0MDEyNn0.47mDreY0riUxPpfu987TNj8Iwd2VCC5jdkWFKW6o3Kg")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connexion Supabase établie")
except Exception as e:
    print(f"❌ ERREUR : Impossible de se connecter à Supabase : {e}")
    exit(1)

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

# ==================== MODÈLES DE DONNÉES ====================

class PseudoCheck(BaseModel):
    pseudo: str

class UserCreate(BaseModel):
    pseudo: str
    skin_id: int = 0

class UpdatePseudoRequest(BaseModel):
    old_pseudo: str
    new_pseudo: str

class UpdateSkinRequest(BaseModel):
    pseudo: str
    skin_id: int

class UpdateUnlockedSkinsRequest(BaseModel):
    pseudo: str
    unlocked_skins: List[int]

class SaveBurpRequest(BaseModel):
    pseudo: str
    score: int
    global_score: int  # ✅ Calculé par Unity
    best_score: int    # ✅ Calculé par Unity

# ==================== ENDPOINTS GESTION UTILISATEURS ====================

@app.post("/check_pseudo")
async def check_pseudo(data: PseudoCheck):
    """Vérifie si un pseudo est disponible"""
    try:
        response = supabase.table("users").select("pseudo").eq("pseudo", data.pseudo).execute()
        
        if len(response.data) > 0:
            return {"available": False, "message": "Pseudo déjà pris"}
        else:
            return {"available": True, "message": "Pseudo disponible"}
    
    except Exception as e:
        print(f"❌ Erreur check_pseudo: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_user")
async def create_user(data: UserCreate):
    """Crée un nouvel utilisateur avec un pseudo unique"""
    try:
        check = supabase.table("users").select("pseudo").eq("pseudo", data.pseudo).execute()
        
        if len(check.data) > 0:
            raise HTTPException(status_code=400, detail="Pseudo déjà pris")
        
        response = supabase.table("users").insert({
            "pseudo": data.pseudo,
            "skin_id": data.skin_id,
            "unlocked_skins": [0],
            "global_score": 0,
            "best_score": 0
        }).execute()
        
        print(f"✅ Utilisateur créé : {data.pseudo}")
        
        return {
            "success": True,
            "user_id": response.data[0]["id"],
            "pseudo": data.pseudo,
            "message": "Utilisateur créé avec succès"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur create_user: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_pseudo")
async def update_pseudo(data: UpdatePseudoRequest):
    """Change le pseudo d'un utilisateur existant"""
    try:
        check_new = supabase.table("users").select("pseudo").eq("pseudo", data.new_pseudo).execute()
        
        if len(check_new.data) > 0:
            raise HTTPException(status_code=400, detail="Nouveau pseudo déjà pris")
        
        check_old = supabase.table("users").select("id").eq("pseudo", data.old_pseudo).execute()
        
        if len(check_old.data) == 0:
            raise HTTPException(status_code=404, detail="Ancien pseudo introuvable")
        
        response = supabase.table("users")\
            .update({"pseudo": data.new_pseudo})\
            .eq("pseudo", data.old_pseudo)\
            .execute()
        
        print(f"✅ Pseudo mis à jour : {data.old_pseudo} → {data.new_pseudo}")
        
        return {
            "success": True,
            "old_pseudo": data.old_pseudo,
            "new_pseudo": data.new_pseudo,
            "message": "Pseudo mis à jour avec succès"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur update_pseudo: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_skin")
async def update_skin(data: UpdateSkinRequest):
    """Met à jour le skin équipé d'un utilisateur"""
    try:
        check = supabase.table("users").select("id").eq("pseudo", data.pseudo).execute()
        
        if len(check.data) == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable")
        
        response = supabase.table("users")\
            .update({"skin_id": data.skin_id})\
            .eq("pseudo", data.pseudo)\
            .execute()
        
        print(f"✅ Skin mis à jour pour {data.pseudo}: skin {data.skin_id}")
        
        return {
            "success": True,
            "message": "Skin mis à jour avec succès"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur update_skin: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_user_skin")
async def get_user_skin(pseudo: str):
    """Récupère le skin équipé d'un utilisateur"""
    try:
        response = supabase.table("users")\
            .select("skin_id")\
            .eq("pseudo", pseudo)\
            .execute()
        
        if len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable")
        
        skin_id = response.data[0].get("skin_id", 0)
        
        print(f"✅ Skin récupéré pour {pseudo}: {skin_id}")
        
        return {
            "success": True,
            "skin_id": skin_id,
            "message": "Skin récupéré avec succès"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur get_user_skin: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_unlocked_skins")
async def update_unlocked_skins(data: UpdateUnlockedSkinsRequest):
    """Met à jour la liste des skins débloqués d'un utilisateur"""
    try:
        check = supabase.table("users").select("id").eq("pseudo", data.pseudo).execute()
        
        if len(check.data) == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable")
        
        response = supabase.table("users")\
            .update({"unlocked_skins": data.unlocked_skins})\
            .eq("pseudo", data.pseudo)\
            .execute()
        
        print(f"✅ Skins débloqués mis à jour pour {data.pseudo}: {data.unlocked_skins}")
        
        return {
            "success": True,
            "message": "Skins débloqués mis à jour avec succès"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur update_unlocked_skins: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_unlocked_skins")
async def get_unlocked_skins(pseudo: str):
    """Récupère la liste des skins débloqués d'un utilisateur"""
    try:
        response = supabase.table("users")\
            .select("unlocked_skins")\
            .eq("pseudo", pseudo)\
            .execute()
        
        if len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable")
        
        unlocked_skins = response.data[0].get("unlocked_skins", [0])
        
        print(f"✅ Skins débloqués récupérés pour {pseudo}: {unlocked_skins}")
        
        return {
            "success": True,
            "unlocked_skins": unlocked_skins,
            "message": "Skins récupérés avec succès"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur get_unlocked_skins: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENDPOINTS POUR LES ROTS ====================

@app.post("/save_burp")
async def save_burp(data: SaveBurpRequest):
    """
    Sauvegarde un rot
    
    ✅ SIMPLIFIÉ : Unity envoie directement global_score et best_score calculés en local
    Le serveur se contente de les stocker
    """
    try:
        # Vérifier que l'utilisateur existe
        user_response = supabase.table("users")\
            .select("id")\
            .eq("pseudo", data.pseudo)\
            .execute()
        
        if len(user_response.data) == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable")
        
        # ✅ Mettre à jour directement avec les valeurs envoyées par Unity
        supabase.table("users").update({
            "global_score": data.global_score,
            "best_score": data.best_score
        }).eq("pseudo", data.pseudo).execute()
        
        print(f"✅ Rot sauvegardé pour {data.pseudo}: {data.score}%")
        print(f"   GlobalScore: {data.global_score}")
        print(f"   BestScore: {data.best_score}")
        
        return {
            "success": True,
            "message": "Rot sauvegardé avec succès",
            "global_score": data.global_score,
            "best_score": data.best_score
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur save_burp: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_user_stats")
async def get_user_stats(pseudo: str):
    """Récupère le GlobalScore et BestScore d'un utilisateur"""
    try:
        response = supabase.table("users")\
            .select("global_score, best_score")\
            .eq("pseudo", pseudo)\
            .execute()
        
        if len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable")
        
        user_data = response.data[0]
        global_score = user_data.get("global_score", 0)
        best_score = user_data.get("best_score", 0)
        
        print(f"✅ Stats récupérées pour {pseudo}: GlobalScore={global_score}, BestScore={best_score}")
        
        return {
            "success": True,
            "global_score": global_score,
            "best_score": best_score,
            "message": "Stats récupérées avec succès"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur get_user_stats: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENDPOINT LEADERBOARD ====================

@app.get("/leaderboard/top{limit}")
async def get_leaderboard(limit: int = 10):
    """Récupère le top N des joueurs avec le meilleur global_score"""
    try:
        print(f"📊 Génération du leaderboard (top {limit})...")
        
        # Récupérer tous les utilisateurs triés par global_score
        response = supabase.table("users")\
            .select("pseudo, global_score, best_score, skin_id")\
            .order("global_score", desc=True)\
            .order("pseudo", desc=False)\
            .limit(limit)\
            .execute()
        
        if len(response.data) == 0:
            print("⚠️ Aucun utilisateur trouvé")
            return {"leaderboard": []}
        
        # Construire le leaderboard
        leaderboard = []
        for idx, user in enumerate(response.data):
            leaderboard.append({
                "rank": idx + 1,
                "pseudo": user["pseudo"],
                "global_score": user.get("global_score", 0),
                "best_score": user.get("best_score", 0),
                "skin_id": user.get("skin_id", 0)
            })
        
        print(f"✅ Leaderboard généré : {len(leaderboard)} entrées")
        
        # Debug : afficher le top 3
        for i in range(min(3, len(leaderboard))):
            entry = leaderboard[i]
            print(f"  🏆 #{entry['rank']} {entry['pseudo']} - {entry['global_score']} pts (Best: {entry['best_score']}%)")
        
        return {"leaderboard": leaderboard}
    
    except Exception as e:
        print(f"❌ Erreur leaderboard: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENDPOINT DÉTECTION DE ROT ====================

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    tmp_path = None
    
    try:
        content = await audio.read()
        if len(content) > 10 * 1024 * 1024:
            print(f"❌ Fichier trop gros : {len(content)} bytes")
            return {"is_rot": False, "confidence": 0.0, "error": "File too large (max 10MB)"}
        
        if len(content) == 0:
            print("❌ Fichier vide reçu")
            return {"is_rot": False, "confidence": 0.0, "error": "Empty audio file"}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"📥 Fichier reçu : {len(content)} bytes")
        
        try:
            data, sr = librosa.load(tmp_path, sr=44100, mono=True, duration=MAX_AUDIO_DURATION)
        except Exception as e:
            print(f"❌ Erreur lors du chargement audio : {e}")
            return {"is_rot": False, "confidence": 0.0, "error": "Invalid audio format"}
        
        audio_duration = len(data) / sr
        print(f"📊 Audio : {len(data)} samples ({audio_duration:.2f}s) à {sr} Hz")
        
        if audio_duration < 0.1:
            print("❌ Audio trop court")
            return {"is_rot": False, "confidence": 0.0, "error": "Audio too short"}
        
        if np.abs(data).max() < 0.001:
            print("❌ Audio vide ou silence total")
            return {"is_rot": False, "confidence": 0.0, "error": "Silent audio"}
        
        num_segments = int(np.ceil(len(data) / SEGMENT_LENGTH))
        print(f"🔪 Découpage en {num_segments} segment(s)")
        
        max_rot_confidence = 0.0
        rot_detected = False
        valid_segments = 0
        
        for i in range(num_segments):
            start = i * SEGMENT_LENGTH
            end = start + SEGMENT_LENGTH
            segment = data[start:end]
            
            if len(segment) < SEGMENT_LENGTH:
                repeats = int(np.ceil(SEGMENT_LENGTH / len(segment)))
                segment = np.tile(segment, repeats)[:SEGMENT_LENGTH]
            
            input_data = np.expand_dims(segment, axis=0).astype(np.float32)
            
            try:
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            except Exception as e:
                print(f"⚠️ Segment {i+1}: Erreur d'inférence : {e}")
                continue
            
            if np.any(np.isnan(output_data)) or np.any(np.isinf(output_data)):
                print(f"⚠️ Segment {i+1}: Valeurs invalides, ignoré")
                continue
            
            valid_segments += 1
            
            if output_data.max() > 1.0 or output_data.min() < 0.0:
                exp_values = np.exp(output_data - np.max(output_data))
                output_data = exp_values / exp_values.sum()
            
            rot_confidence = float(output_data[1])
            max_rot_confidence = max(max_rot_confidence, rot_confidence)
            
            print(f"  Segment {i+1}/{num_segments} ({start/sr:.2f}-{end/sr:.2f}s): "
                  f"Background={output_data[0]:.2%}, Rot={rot_confidence:.2%}")
            
            if rot_confidence >= ROT_THRESHOLD:
                rot_detected = True
                print(f"  🎉 ROT DÉTECTÉ dans segment {i+1} !")
        
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
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"🗑️ Fichier temporaire supprimé")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer le fichier : {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True, "database": "connected"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage du serveur...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
