"""
Voice Stylist Adapter for Agentic Pipeline
Bridges the existing Flask API with VoicePersonalStylist
"""

import os
import requests
from typing import Dict, List, Optional
from voice_stylist import VoicePersonalStylist


class ClosetVoiceStylist(VoicePersonalStylist):
    """
    Adapter for VoicePersonalStylist that works with router.py catalog endpoints.
    
    Key differences from base class:
    - Uses email-based auth instead of JWT tokens
    - Maps field names from router.py/Supabase format
    - Calls /catalog/collection
    """
    
    def __init__(
        self,
        user_email: str,
        closet_api_url: str = "http://localhost:8000",
        openai_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    ):
        """
        Initialize adapter.
        
        Args:
            user_email: User's email address
            closet_api_url: Base URL of Flask API (default: http://localhost:5001/api)
            openai_api_key: OpenAI API key
            elevenlabs_api_key: ElevenLabs API key
            voice_id: ElevenLabs voice ID
        """
        # Initialize parent without auth_token
        super().__init__(
            closet_api_url=closet_api_url,
            auth_token=None,  # Not used
            openai_api_key=openai_api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id
        )
        
        # Store email for API calls (not used by catalog endpoint)
        self.user_email = user_email
        self.headers = {}  # No auth headers needed
    
    def fetch_user_closet(self) -> List[Dict]:
        """Fetch closet items from router.py catalog collection."""
        try:
            response = requests.get(
                f"{self.closet_api_url}/catalog/collection",
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            collection = data.get("collection", [])

            mapped_items: List[Dict] = []
            for job in collection:
                for crop in job.get("crops", []) or []:
                    metadata = crop.get("metadata") or {}
                    colors = metadata.get("color") or []
                    if isinstance(colors, list):
                        color_hex = colors[0] if colors else "Unknown"
                        color_hex = color_hex if isinstance(color_hex, str) else "Unknown"
                    else:
                        color_hex = str(colors)
                    mapped_items.append({
                        "id": crop.get("id") or f"{job.get('id')}_{crop.get('label')}",
                        "category": crop.get("label", "Unknown"),
                        "color_hex": color_hex,
                        "material_inference": metadata.get("material", "Unknown"),
                        "occasion_vibe": metadata.get("style", "Unknown"),
                        "seasonality": metadata.get("season", "Unknown"),
                        "brand_guess": metadata.get("brand", "Generic"),
                        "image_path": crop.get("crop_url") or crop.get("generated_url"),
                        "_original": crop,
                    })

            self.user_closet = mapped_items
            
            print(f"✓ Loaded {len(self.user_closet)} items from closet")
            
            # Automatically analyze closet to infer user profile
            self._analyze_closet_profile()
            
            return self.user_closet
            
        except Exception as e:
            print(f"✗ Error fetching closet: {e}")
            return []
    
    def fetch_upload_history(self) -> List[Dict]:
        """Fetch upload history from router.py catalog collection."""
        try:
            response = requests.get(
                f"{self.closet_api_url}/catalog/collection",
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            collection = data.get("collection", [])
            uploads = []
            for job in collection:
                uploads.append({
                    "id": job.get("id"),
                    "global_analysis": job.get("scene_analysis") or {},
                })
            print(f"✓ Loaded {len(uploads)} upload histories")
            return uploads
            
        except Exception as e:
            print(f"✗ Error fetching upload history: {e}")
            return []
    
    def _color_name_to_hex(self, color_name: Optional[str]) -> str:
        """
        Convert color name to hex code.
        
        This is a basic mapping - expand as needed based on your data.
        """
        if not color_name:
            return "#808080"  # Gray default
        
        color_name = color_name.lower().strip()
        
        # Common color mappings
        color_map = {
            # Basics
            'black': '#000000',
            'white': '#FFFFFF',
            'gray': '#808080',
            'grey': '#808080',
            
            # Primary colors
            'red': '#FF0000',
            'blue': '#0000FF',
            'green': '#008000',
            'yellow': '#FFFF00',
            
            # Secondary colors
            'orange': '#FFA500',
            'purple': '#800080',
            'pink': '#FFC0CB',
            
            # Browns & neutrals
            'brown': '#A52A2A',
            'beige': '#F5F5DC',
            'cream': '#FFFDD0',
            'khaki': '#C3B091',
            'tan': '#D2B48C',
            
            # Blues
            'navy': '#000080',
            'navy blue': '#000080',
            'dark blue': '#00008B',
            'light blue': '#ADD8E6',
            'sky blue': '#87CEEB',
            'royal blue': '#4169E1',
            'teal': '#008080',
            'turquoise': '#40E0D0',
            'cyan': '#00FFFF',
            'indigo': '#4B0082',
            'denim': '#1560BD',
            
            # Greens
            'olive': '#808000',
            'lime': '#00FF00',
            'mint': '#98FF98',
            'forest green': '#228B22',
            'emerald': '#50C878',
            
            # Reds & pinks
            'maroon': '#800000',
            'burgundy': '#800020',
            'coral': '#FF7F50',
            'salmon': '#FA8072',
            'magenta': '#FF00FF',
            'fuchsia': '#FF00FF',
            
            # Purples
            'lavender': '#E6E6FA',
            'violet': '#EE82EE',
            'plum': '#DDA0DD',
            
            # Metallics
            'silver': '#C0C0C0',
            'gold': '#FFD700',
            
            # Grays
            'charcoal': '#36454F',
            'slate': '#708090',
            'ash': '#B2BEB5',
            
            # Others
            'ivory': '#FFFFF0',
            'ecru': '#C2B280',
            'taupe': '#483C32',
            'rust': '#B7410E',
            'mustard': '#FFDB58',
            'olive drab': '#6B8E23',
        }
        
        # Try exact match first
        if color_name in color_map:
            return color_map[color_name]
        
        # Try partial match (e.g., "dark red" contains "red")
        for name, hex_code in color_map.items():
            if name in color_name or color_name in name:
                return hex_code
        
        # Default to gray if no match
        print(f"⚠️  Unknown color '{color_name}', defaulting to gray")
        return '#808080'


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import os
    
    print("="*60)
    print("  Voice Personal Stylist - Agentic Pipeline Integration")
    print("="*60)
    print()
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY=your_key")
        exit(1)
    
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ERROR: ELEVENLABS_API_KEY not set")
        print("Set it with: export ELEVENLABS_API_KEY=your_key")
        exit(1)
    
    # Get user email
    user_email = input("Enter user email: ").strip()
    if not user_email:
        print("❌ ERROR: Email is required")
        exit(1)
    
    # Initialize stylist with user email
    print(f"\nInitializing stylist for {user_email}...")
    stylist = ClosetVoiceStylist(
        user_email=user_email,
        closet_api_url="http://localhost:5001/api",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="EXAVITQu4vr4xnSDxMaL"  # Bella voice (warm, friendly)
    )
    
    # Start voice styling session
    print("\nStarting voice styling session...")
    print("Speak clearly into your microphone when prompted.\n")
    stylist.start_styling_session()
