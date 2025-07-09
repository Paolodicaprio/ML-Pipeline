#!/usr/bin/env python3
"""
Script pour exécuter tous les tests unitaires.
"""
import unittest
import sys
import os

if __name__ == "__main__":
    # Découvrir et exécuter tous les tests dans le répertoire courant
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__))
    
    # Exécuter les tests
    result = unittest.TextTestRunner().run(test_suite)
    
    # Sortir avec un code d'erreur si des tests ont échoué
    sys.exit(not result.wasSuccessful())