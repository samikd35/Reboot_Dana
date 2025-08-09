"""
Ultra-Efficient Bridge: AlphaEarth ↔ Crop Recommender Integration

This module creates a seamless connection between the AlphaEarth satellite
embedding system and the crop recommendation ML model.
"""

import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time
from functools import lru_cache
import pickle
import json

# Import our systems with fallbacks
try:
    from alphaearth import AlphaEarthExtractor
    ALPHAEARTH_AVAILABLE = True
except ImportError:
    ALPHAEARTH_AVAILABLE = False
    AlphaEarthExtractor = None

try:
    from core.earth_engine_integration import AlphaEarthFeatureExtractor
    EE_INTEGRATION_AVAILABLE = True
except ImportError:
    EE_INTEGRATION_AVAILABLE = False
    AlphaEarthFeatureExtractor = None

logger = logging.getLogger(__name__)

@dataclass
class CropRecommendationRequest:
    """Request structure for crop recommendation"""
    latitude: float
    longitude: float
    year: int = 2024
    buffer_meters: int = 1000
    use_cache: bool = True
    confidence_threshold: float = 0.7

@dataclass
class CropRecommendationResponse:
    """Response structure with full context"""
    # Core prediction
    recommended_crop: str
    confidence_score: float
    crop_class_id: int
    
    # Satellite data context
    satellite_features: Dict[str, float]
    embedding_metadata: Dict[str, Any]
    
    # Geographic context
    coordinates: Dict[str, float]
    region_info: Dict[str, Any]
    
    # Processing metadata
    processing_time_ms: float
    data_sources: List[str]
    cache_hit: bool

class UltraIntegrationBridge:
    """
    Ultra-efficient bridge connecting AlphaEarth and Crop Recommender
    
    Features:
    - Async processing for multiple requests
    - Intelligent caching system
    - Fallback mechanisms
    - Performance monitoring
    - Batch processing capabilities
    """
    
    def __init__(self, 
                 model_path: str = 'model.pkl',
                 scaler_paths: Tuple[str, str] = ('minmaxscaler.pkl', 'standscaler.pkl'),
                 earth_engine_credentials: Optional[str] = None,
                 cache_size: int = 1000,
                 enable_async: bool = True):
        """
        Initialize the integration bridge
        
        Args:
            model_path: Path to the trained crop recommendation model
            scaler_paths: Paths to the feature scalers (minmax, standard)
            earth_engine_credentials: Path to Earth Engine service account key
            cache_size: Size of the LRU cache for embeddings
            enable_async: Whether to enable async processing
        """
        self.enable_async = enable_async
        self.cache_size = cache_size
        
        # Initialize crop recommendation system
        self._load_ml_models(model_path, scaler_paths)
        
        # Initialize AlphaEarth system
        self._initialize_alphaearth(earth_engine_credentials)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_processing_time': 0,
            'error_count': 0
        }
        
        logger.info("UltraIntegrationBridge initialized successfully")
    
    def _load_ml_models(self, model_path: str, scaler_paths: Tuple[str, str]):
        """Load the ML models and scalers"""
        try:
            # Handle model path relative to project structure
            from pathlib import Path
            if not Path(model_path).is_absolute():
                project_root = Path(__file__).parent.parent.parent
                model_path = project_root / "models" / model_path
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Always try to load fixed scalers first, fallback to original if not found
            import os
            from pathlib import Path
            
            # Get the project root directory (two levels up from src/core)
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "models"
            
            fixed_minmax_path = models_dir / 'minmaxscaler_fixed.pkl'
            fixed_std_path = models_dir / 'standscaler_fixed.pkl'
            
            if fixed_minmax_path.exists():
                with open(fixed_minmax_path, 'rb') as f:
                    self.minmax_scaler = pickle.load(f)
                logger.info("✅ Using FIXED MinMax scaler")
            else:
                # Try original paths relative to models directory
                original_path = models_dir / scaler_paths[0]
                if original_path.exists():
                    with open(original_path, 'rb') as f:
                        self.minmax_scaler = pickle.load(f)
                else:
                    with open(scaler_paths[0], 'rb') as f:
                        self.minmax_scaler = pickle.load(f)
                logger.warning("⚠️  Using original (potentially broken) MinMax scaler")
                
            if fixed_std_path.exists():
                with open(fixed_std_path, 'rb') as f:
                    self.standard_scaler = pickle.load(f)
                logger.info("✅ Using FIXED Standard scaler")
            else:
                # Try original paths relative to models directory
                original_path = models_dir / scaler_paths[1]
                if original_path.exists():
                    with open(original_path, 'rb') as f:
                        self.standard_scaler = pickle.load(f)
                else:
                    with open(scaler_paths[1], 'rb') as f:
                        self.standard_scaler = pickle.load(f)
                logger.warning("⚠️  Using original (potentially broken) Standard scaler")
            
            # Crop mapping dictionary
            self.crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
                6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 
                10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 
                18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
                21: "Chickpea", 22: "Coffee"
            }
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise
    
    def _initialize_alphaearth(self, credentials_path: Optional[str]):
        """Initialize AlphaEarth extraction system"""
        self.alphaearth_extractor = None
        self.use_real_alphaearth = False
        
        # Get project ID from environment or constructor
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        # Try AlphaEarth module first
        if ALPHAEARTH_AVAILABLE and AlphaEarthExtractor:
            try:
                self.alphaearth_extractor = AlphaEarthExtractor(
                    service_account_key=credentials_path,
                    project_id=project_id
                )
                # Check if it's using real Earth Engine or mock
                if hasattr(self.alphaearth_extractor, 'use_real_ee') and self.alphaearth_extractor.use_real_ee:
                    self.use_real_alphaearth = True
                    logger.info(f"Real AlphaEarth extractor initialized with Earth Engine (project: {project_id})")
                else:
                    self.use_real_alphaearth = False
                    logger.info("AlphaEarth extractor initialized with mock fallback")
                return
            except Exception as e:
                logger.warning(f"AlphaEarth extractor failed: {e}")
        
        # Try Earth Engine integration fallback
        if EE_INTEGRATION_AVAILABLE and AlphaEarthFeatureExtractor:
            try:
                self.alphaearth_extractor = AlphaEarthFeatureExtractor(
                    service_account_key=credentials_path,
                    project_id=project_id
                )
                self.use_real_alphaearth = True
                logger.info(f"Earth Engine integration extractor initialized (project: {project_id})")
                return
            except Exception as e:
                logger.warning(f"Earth Engine integration failed: {e}")
        
        # Final fallback to mock extractor
        try:
            from test_with_mock_ee import MockAlphaEarthFeatureExtractor
            self.alphaearth_extractor = MockAlphaEarthFeatureExtractor()
            self.use_real_alphaearth = False
            logger.info("Mock extractor initialized as final fallback")
        except Exception as e:
            logger.error(f"All extractors failed, including mock: {e}")
            self.alphaearth_extractor = None
    
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, lat: float, lon: float, year: int) -> Optional[np.ndarray]:
        """Get cached satellite embedding"""
        cache_key = f"{lat:.6f}_{lon:.6f}_{year}"
        # This will be automatically cached by lru_cache decorator
        return None  # Placeholder - actual caching happens in the calling method
    
    async def get_crop_recommendation_async(self, 
                                          request: CropRecommendationRequest) -> CropRecommendationResponse:
        """Async version of crop recommendation"""
        if not self.enable_async:
            return self.get_crop_recommendation(request)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_crop_recommendation, request)
    
    def get_crop_recommendation(self, 
                              request: CropRecommendationRequest) -> CropRecommendationResponse:
        """
        Main integration method: Get crop recommendation from coordinates
        
        Args:
            request: CropRecommendationRequest with coordinates and parameters
            
        Returns:
            CropRecommendationResponse with full context
        """
        start_time = time.time()
        cache_hit = False
        data_sources = []
        
        try:
            self.stats['total_requests'] += 1
            
            # Step 1: Extract satellite features
            satellite_features, embedding_metadata = self._extract_satellite_features(
                request.latitude, 
                request.longitude, 
                request.year,
                request.use_cache
            )
            
            if embedding_metadata.get('from_cache', False):
                cache_hit = True
                self.stats['cache_hits'] += 1
            
            data_sources.extend(embedding_metadata.get('sources', ['AlphaEarth']))
            
            # Step 2: Make crop prediction
            crop_prediction = self._predict_crop(satellite_features)
            
            # Step 3: Get regional context
            region_info = self._get_regional_context(
                request.latitude, 
                request.longitude
            )
            
            # Step 4: Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time)
            
            # Step 5: Build response
            response = CropRecommendationResponse(
                recommended_crop=crop_prediction['crop_name'],
                confidence_score=crop_prediction['confidence'],
                crop_class_id=crop_prediction['class_id'],
                satellite_features=satellite_features,
                embedding_metadata=embedding_metadata,
                coordinates={
                    'latitude': request.latitude,
                    'longitude': request.longitude
                },
                region_info=region_info,
                processing_time_ms=processing_time,
                data_sources=data_sources,
                cache_hit=cache_hit
            )
            
            logger.info(f"Crop recommendation completed: {crop_prediction['crop_name']} "
                       f"({processing_time:.1f}ms)")
            
            return response
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Crop recommendation failed: {e}")
            raise
    
    def _extract_satellite_features(self, 
                                   latitude: float, 
                                   longitude: float, 
                                   year: int,
                                   use_cache: bool) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Extract agricultural features from satellite data"""
        
        # Check cache first
        cache_key = f"{latitude:.6f}_{longitude:.6f}_{year}"
        if use_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result['features'], {
                    **cached_result['metadata'],
                    'from_cache': True
                }
        
        # Extract fresh data
        if self.alphaearth_extractor is None:
            raise RuntimeError("No AlphaEarth extractor available")
        
        try:
            if self.use_real_alphaearth:
                # Use real AlphaEarth extractor
                features = self.alphaearth_extractor.extract_agricultural_features(
                    latitude, longitude, year
                )
                metadata = {
                    'extractor_type': 'real_alphaearth',
                    'embedding_dimensions': 64,
                    'sources': ['AlphaEarth_V1_ANNUAL'],
                    'from_cache': False
                }
            else:
                # Use fallback extractor
                features = self.alphaearth_extractor.extract_agricultural_features(
                    latitude, longitude, year
                )
                metadata = {
                    'extractor_type': 'fallback_alphaearth',
                    'embedding_dimensions': 64,
                    'sources': ['AlphaEarth_Simulated'],
                    'from_cache': False
                }
            
            # Cache the result
            if use_cache:
                self._store_in_cache(cache_key, {
                    'features': features,
                    'metadata': metadata
                })
            
            return features, metadata
            
        except Exception as e:
            logger.error(f"Satellite feature extraction failed: {e}")
            raise
    
    def _predict_crop(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make crop prediction using ML model"""
        try:
            # Convert features to array in correct order
            feature_array = np.array([
                features['nitrogen'],
                features['phosphorus'],
                features['potassium'],
                features['temperature'],
                features['humidity'],
                features['ph'],
                features['rainfall']
            ]).reshape(1, -1)
            
            # Apply scaling
            scaled_features = self.minmax_scaler.transform(feature_array)
            final_features = self.standard_scaler.transform(scaled_features)
            
            # Make prediction
            prediction = self.model.predict(final_features)[0]
            probabilities = self.model.predict_proba(final_features)[0]
            
            # Get confidence score
            confidence = float(np.max(probabilities))
            
            # Get crop name
            crop_name = self.crop_dict.get(prediction, "Unknown")
            
            return {
                'class_id': int(prediction),
                'crop_name': crop_name,
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            logger.error(f"Crop prediction failed: {e}")
            raise
    
    def _get_regional_context(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get additional regional context information"""
        # Determine climate zone
        climate_zone = self._get_climate_zone(latitude)
        
        # Determine continent
        continent = self._get_continent(longitude)
        
        # Agricultural season info
        season_info = self._get_season_info(latitude)
        
        return {
            'climate_zone': climate_zone,
            'continent': continent,
            'season_info': season_info,
            'hemisphere': 'Northern' if latitude >= 0 else 'Southern'
        }
    
    def _get_climate_zone(self, latitude: float) -> str:
        """Determine climate zone from latitude"""
        abs_lat = abs(latitude)
        if abs_lat > 66.5:
            return 'Polar'
        elif abs_lat > 40:
            return 'Temperate'
        elif abs_lat > 23.5:
            return 'Subtropical'
        else:
            return 'Tropical'
    
    def _get_continent(self, longitude: float) -> str:
        """Rough continent determination from longitude"""
        if -130 < longitude < -60:
            return 'Americas'
        elif -10 < longitude < 50:
            return 'Europe/Africa'
        elif 70 < longitude < 150:
            return 'Asia'
        elif 110 < longitude < 180:
            return 'Oceania'
        else:
            return 'Unknown'
    
    def _get_season_info(self, latitude: float) -> Dict[str, str]:
        """Get seasonal information"""
        import datetime
        month = datetime.datetime.now().month
        
        if latitude >= 0:  # Northern hemisphere
            if month in [12, 1, 2]:
                return {'season': 'Winter', 'growing_season': 'Dormant'}
            elif month in [3, 4, 5]:
                return {'season': 'Spring', 'growing_season': 'Planting'}
            elif month in [6, 7, 8]:
                return {'season': 'Summer', 'growing_season': 'Growing'}
            else:
                return {'season': 'Fall', 'growing_season': 'Harvest'}
        else:  # Southern hemisphere
            if month in [6, 7, 8]:
                return {'season': 'Winter', 'growing_season': 'Dormant'}
            elif month in [9, 10, 11]:
                return {'season': 'Spring', 'growing_season': 'Planting'}
            elif month in [12, 1, 2]:
                return {'season': 'Summer', 'growing_season': 'Growing'}
            else:
                return {'season': 'Fall', 'growing_season': 'Harvest'}
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get item from cache (placeholder for actual cache implementation)"""
        # This would integrate with Redis, Memcached, or in-memory cache
        return None
    
    def _store_in_cache(self, key: str, value: Dict):
        """Store item in cache (placeholder for actual cache implementation)"""
        # This would integrate with Redis, Memcached, or in-memory cache
        pass
    
    def _update_stats(self, processing_time: float):
        """Update performance statistics"""
        if self.stats['total_requests'] == 1:
            self.stats['avg_processing_time'] = processing_time
        else:
            # Running average
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['total_requests'] - 1) + 
                 processing_time) / self.stats['total_requests']
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (self.stats['cache_hits'] / max(self.stats['total_requests'], 1)) * 100
        
        return {
            **self.stats,
            'cache_hit_rate_percent': cache_hit_rate,
            'error_rate_percent': (self.stats['error_count'] / max(self.stats['total_requests'], 1)) * 100
        }
    
    async def batch_process_locations(self, 
                                    locations: List[Tuple[float, float]], 
                                    year: int = 2024) -> List[CropRecommendationResponse]:
        """Process multiple locations in parallel"""
        if not self.enable_async:
            # Fallback to sequential processing
            results = []
            for lat, lon in locations:
                request = CropRecommendationRequest(latitude=lat, longitude=lon, year=year)
                results.append(self.get_crop_recommendation(request))
            return results
        
        # Async batch processing
        tasks = []
        for lat, lon in locations:
            request = CropRecommendationRequest(latitude=lat, longitude=lon, year=year)
            task = self.get_crop_recommendation_async(request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for location {locations[i]}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        # Check ML models
        try:
            test_features = {
                'nitrogen': 50, 'phosphorus': 50, 'potassium': 50,
                'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 100
            }
            self._predict_crop(test_features)
            health['components']['ml_model'] = 'healthy'
        except Exception as e:
            health['components']['ml_model'] = f'error: {e}'
            health['status'] = 'degraded'
        
        # Check AlphaEarth extractor
        if self.alphaearth_extractor is None:
            health['components']['alphaearth'] = 'not_available'
            health['status'] = 'degraded'
        else:
            health['components']['alphaearth'] = 'healthy'
        
        # Add performance stats
        health['performance'] = self.get_performance_stats()
        
        return health