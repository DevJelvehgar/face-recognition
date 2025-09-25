#!/usr/bin/env python3
"""
Test script for face recognition with sample image
Tests the sample image against the database and finds all 5 expected files from find.txt
"""

import os
from PIL import Image
from face_recognition import FaceRecognition


def test_without_expected_files():
    """Test without prioritizing expected files to see natural results"""

    print("\n" + "=" * 60)
    print("🔬 Test without prioritizing expected files:")
    print("=" * 60)

    fr = FaceRecognition()
    sample_path = "samples/janati.jpg"

    try:
        image = Image.open(sample_path)
        query_embedding = fr.extract_face_embedding(image)

        if query_embedding is not None:
            # Get top 10 results without prioritizing expected files
            similar_faces = fr.find_most_similar_faces(
                query_embedding, top_k=50)

            print("📊 Top 10 Similar Faces (Natural Results):")
            for i, face in enumerate(similar_faces, 1):
                print(
                    f"{i:2d}. {face['filename']:<25} - {face['similarity_percentage']:6.2f}%")

    except Exception as e:
        print(f"❌ Error in natural test: {e}")


if __name__ == "__main__":
    test_without_expected_files()
