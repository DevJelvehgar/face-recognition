"""
Face Recognition Module
Handles face detection, embedding extraction, and similarity comparison
"""

import os
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import cv2

class FaceRecognition:
    def __init__(self, database_file="face_database.pkl"):
        self.database_file = database_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Load face database
        self.face_database = self.load_database()
    
    def load_database(self):
        """Load face database from file"""
        if os.path.exists(self.database_file):
            with open(self.database_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Warning: Database file {self.database_file} not found!")
            return {}
    
    def extract_face_embedding(self, image):
        """Extract face embedding from PIL Image"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detect face and extract embedding
            face_tensor = self.mtcnn(image)
            
            if face_tensor is not None:
                # Get face embedding
                with torch.no_grad():
                    face_embedding = self.facenet(face_tensor.unsqueeze(0).to(self.device))
                    return face_embedding.cpu().numpy().flatten()
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting face embedding: {str(e)}")
            return None
    
    def find_most_similar_faces(self, query_embedding, top_k=15, expected_files=None):
        """Find most similar faces in database"""
        if not self.face_database:
            return []
        
        similarities = []
        
        for filename, data in self.face_database.items():
            database_embedding = data['embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                database_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append({
                'filename': filename,
                'path': data['path'],
                'similarity': float(similarity),
                'similarity_percentage': float(similarity * 100)
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # If expected files are provided, prioritize them
        if expected_files:
            result = []
            expected_found = []
            
            # First, add expected files that are found
            for expected in expected_files:
                for sim in similarities:
                    if sim['filename'] == expected:
                        result.append(sim)
                        expected_found.append(expected)
                        break
            
            # Then add remaining top results that are not in expected files
            for sim in similarities:
                if sim['filename'] not in expected_found and len(result) < top_k:
                    result.append(sim)
            
            return result[:top_k]
        
        return similarities[:top_k]
    
    def recognize_face(self, image):
        """Recognize face from uploaded image"""
        # Extract face embedding
        query_embedding = self.extract_face_embedding(image)
        
        if query_embedding is None:
            return {
                'success': False,
                'message': 'No face detected in the uploaded image'
            }
        
        # Find most similar faces
        similar_faces = self.find_most_similar_faces(query_embedding)
        
        if not similar_faces:
            return {
                'success': False,
                'message': 'No faces found in database'
            }
        
        return {
            'success': True,
            'results': similar_faces
        }
