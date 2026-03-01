"""
AI Voice Agent Personal Stylist
Integrates with Digital Closet API to provide outfit recommendations
Uses ElevenLabs for natural voice synthesis and OpenAI for intelligent conversation
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import requests
import io
from dotenv import load_dotenv

# Voice/Speech libraries
from google import genai
from google.genai import types
import speech_recognition as sr
from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine


class VoicePersonalStylist:
    """AI-powered voice agent that acts as a personal stylist."""
    
    def __init__(
        self,
        closet_api_url: str = "http://localhost:8000",
        auth_token: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        voice_id: str = "EXAVITQu4vr4xnSDxMaL"  # Default: Bella (warm, friendly female voice)
    ):
        """
        Initialize the voice stylist agent.
        
        Args:
            closet_api_url: Base URL of your Digital Closet API
            auth_token: JWT token for authenticated API calls (optional, not needed for email-based auth)
            elevenlabs_api_key: ElevenLabs API key for voice synthesis and speech-to-text
            voice_id: ElevenLabs voice ID (default is Bella)
        """
        # Load environment variables from .env if present
        load_dotenv()
        self.closet_api_url = closet_api_url
        self.auth_token = auth_token
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        
        # Initialize Gemini client (for conversation and analysis)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        
        # Initialize ElevenLabs client
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
        self.voice_id = voice_id
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise on initialization
        with self.microphone as source:
            print("Calibrating for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        
        # Conversation state
        self.user_closet = []
        self.conversation_history = []
        
        # SCENARIO 18: Track past recommendations to avoid repetition
        self.recommendation_history = []  # List of {outfit, timestamp, event_type}
        
        self.user_context = {
            # Event context
            "event_type": None,
            "event_formality": None,
            "weather": None,
            "occasion_details": None,
            
            # SCENARIO 10: Multi-event support
            "is_multi_event": False,
            "events": [],  # [{type, formality, time}]
            
            # Emotional/psychological
            "mood": None,
            "desired_feeling": None,
            "outfit_goals": None,
            "body_concerns": [],
            
            # Style (from closet analysis)
            "inferred_style": None,
            "personality_traits": [],
            "style_preferences": [],
            "color_preferences": [],
            "formality_preference": None,
            "shopping_style": None,
            "lifestyle": None,
            
            # Historical context (from VLM)
            "vlm_history": [],
            "typical_scenes": [],
            "typical_activities": [],
            
            # Meta
            "profile_confidence": None
        }
        
    # ==========================================
    # CLOSET API INTEGRATION
    # ==========================================
    
    def _play_mellow_music(self, duration_ms: int = 1200) -> None:
        """Play a short mellow tone as a thinking cue."""
        try:
            tone = Sine(220).to_audio_segment(duration=duration_ms).apply_gain(-26)
            tone = tone.fade_in(150).fade_out(300)
            play(tone)
        except Exception:
            # Non-critical: skip if audio playback fails
            pass
    
    def _thinking(self, message: str) -> None:
        """Speak a gentle thinking message with mellow cue."""
        print(f"💭 {message}")
        self._play_mellow_music()
        self.speak(message)
    
    def fetch_user_closet(self) -> List[Dict]:
        """Fetch all items from catalog collection via API and map to closet schema."""
        self._thinking("Collecting your profile from your closet.")
        try:
            response = requests.get(
                f"{self.closet_api_url}/catalog/collection",
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            collection = data.get("collection", [])

            items: List[Dict] = []
            for job in collection:
                for crop in job.get("crops", []) or []:
                    metadata = crop.get("metadata") or {}
                    colors = metadata.get("color") or []
                    if isinstance(colors, list):
                        color_hex = colors[0] if colors else "Unknown"
                        color_hex = color_hex if isinstance(color_hex, str) else "Unknown"
                    else:
                        color_hex = str(colors)
                    items.append({
                        "id": crop.get("id") or f"{job.get('id')}_{crop.get('label')}",
                        "category": crop.get("label", "Unknown"),
                        "color_hex": color_hex,
                        "occasion_vibe": metadata.get("style", "Unknown"),
                        "seasonality": metadata.get("season", "Unknown"),
                        "material_inference": metadata.get("material", "Unknown"),
                        "brand_guess": metadata.get("brand", "Generic"),
                        "image_path": crop.get("crop_url") or crop.get("generated_url"),
                    })

            self.user_closet = items
            print(f"✓ Loaded {len(self.user_closet)} items from catalog")

            # Automatically analyze closet to infer user profile
            self._analyze_closet_profile()

            return self.user_closet
        except Exception as e:
            print(f"✗ Error fetching closet: {e}")
            return []
    
    def fetch_upload_history(self) -> List[Dict]:
        """Fetch job-level scene analyses from catalog collection."""
        self._thinking("Fetching your recent style moments.")
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
            return uploads
        except Exception as e:
            print(f"✗ Error fetching upload history: {e}")
            return []
    
    def _analyze_closet_profile(self) -> None:
        """
        Automatically analyze user's closet to infer their style, personality, and preferences.
        Uses GPT-4 to extract patterns from their existing items.
        """
        if not self.user_closet:
            return
        
        self._thinking("Working to make you feel beautiful with a personalized profile.")
        
        # Gather statistics from closet
        categories = {}
        colors = {}
        occasions = {}
        seasonality = {}
        brands = {}
        materials = {}
        
        for item in self.user_closet:
            # Count categories
            cat = item.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count colors
            color = item.get("color_hex", "Unknown")
            colors[color] = colors.get(color, 0) + 1
            
            # Count occasions
            occasion = item.get("occasion_vibe", "Unknown")
            occasions[occasion] = occasions.get(occasion, 0) + 1
            
            # Count seasonality
            season = item.get("seasonality", "Unknown")
            seasonality[season] = seasonality.get(season, 0) + 1
            
            # Count brands
            brand = item.get("brand_guess", "Unknown")
            if brand != "Generic":
                brands[brand] = brands.get(brand, 0) + 1
            
            # Count materials
            material = item.get("material_inference", "Unknown")
            materials[material] = materials.get(material, 0) + 1
        
        # Fetch VLM analysis from upload history
        uploads = self.fetch_upload_history()
        vlm_contexts = [u.get("global_analysis", {}) for u in uploads if u.get("global_analysis")]
        
        # Store VLM history in context for recommendation use
        self.user_context["vlm_history"] = vlm_contexts
        
        # Build comprehensive profile analysis prompt
        analysis_prompt = f"""Analyze this user's fashion profile based on their digital closet:

CLOSET STATISTICS:
- Total items: {len(self.user_closet)}
- Categories: {json.dumps(categories, indent=2)}
- Color distribution: {json.dumps(colors, indent=2)}
- Occasion preferences: {json.dumps(occasions, indent=2)}
- Seasonality: {json.dumps(seasonality, indent=2)}
- Brands: {json.dumps(brands, indent=2)}
- Materials: {json.dumps(materials, indent=2)}

VLM ANALYSIS FROM PAST UPLOADS (Scene contexts & style observations):
{json.dumps(vlm_contexts, indent=2)}

Based on this data, infer:
1. Overall style (e.g., Minimalist, Bold, Classic, Streetwear, Bohemian, Preppy)
2. Personality traits (e.g., adventurous, conservative, creative, practical)
3. Color preferences (which colors they gravitate toward)
4. Formality level (casual, business, formal mix)
5. Shopping habits (luxury brands vs. affordable, trendy vs. timeless)
6. Body type considerations (if inferable from VLM scene contexts)
7. Lifestyle indicators (active, professional, social, home-focused)
8. Typical scene contexts (where they usually go based on VLM: parks, offices, cafes, events)
9. Activity patterns (what they typically do: casual outings, professional work, social gatherings)

Return a JSON object:
{{
    "inferred_style": "primary style description",
    "personality_traits": ["trait1", "trait2", "trait3"],
    "favorite_colors": ["#HEX1", "#HEX2"],
    "formality_preference": "casual/business/formal/mixed",
    "shopping_style": "description",
    "lifestyle": "description",
    "typical_scenes": ["scene1", "scene2"],
    "typical_activities": ["activity1", "activity2"],
    "confidence_level": "how confident are we in this analysis (low/medium/high)"
}}
"""
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part(text=analysis_prompt)],
                config=types.GenerateContentConfig(
                    temperature=0.3  # Lower temperature for more consistent analysis
                )
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to parse JSON (handle markdown code blocks if present)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                profile = json.loads(json_match.group())
            else:
                profile = json.loads(response_text)
            
            # Update user_context with inferred profile
            self.user_context["inferred_style"] = profile.get("inferred_style")
            self.user_context["personality_traits"] = profile.get("personality_traits", [])
            self.user_context["color_preferences"] = profile.get("favorite_colors", [])
            self.user_context["formality_preference"] = profile.get("formality_preference")
            self.user_context["shopping_style"] = profile.get("shopping_style")
            self.user_context["lifestyle"] = profile.get("lifestyle")
            self.user_context["typical_scenes"] = profile.get("typical_scenes", [])
            self.user_context["typical_activities"] = profile.get("typical_activities", [])
            self.user_context["profile_confidence"] = profile.get("confidence_level", "medium")
            
            print(f"✓ Analyzed closet profile: {profile.get('inferred_style')}")
            print(f"  Personality: {', '.join(profile.get('personality_traits', []))}")
            print(f"  Typical scenes: {', '.join(profile.get('typical_scenes', []))}")
            print(f"  Confidence: {profile.get('confidence_level')}")
            
        except Exception as e:
            print(f"✗ Profile analysis error: {e}")
            # Continue without profile - will ask questions instead
    
    def filter_closet_items(
        self,
        category: Optional[str] = None,
        seasonality: Optional[str] = None,
        occasion_vibe: Optional[str] = None,
        color_hex: Optional[str] = None
    ) -> List[Dict]:
        """Filter closet items based on criteria."""
        filtered = self.user_closet
        
        if category:
            filtered = [item for item in filtered if item.get("category") == category]
        if seasonality:
            filtered = [item for item in filtered if item.get("seasonality") in [seasonality, "All-Season"]]
        if occasion_vibe:
            filtered = [item for item in filtered if item.get("occasion_vibe") == occasion_vibe]
        if color_hex:
            filtered = [item for item in filtered if item.get("color_hex") == color_hex]
        
        return filtered
    
    # ==========================================
    # VOICE INTERACTION
    # ==========================================
    
    def speak(self, text: str, save_path: Optional[str] = None) -> None:
        """
        Convert text to speech using ElevenLabs and play it.
        
        Args:
            text: Text to speak
            save_path: Optional path to save audio file
        """
        try:
            print(f"🎙️ Stylist: {text}")
            
            # Generate audio using ElevenLabs
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.5,
                    use_speaker_boost=True
                )
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)
            
            # Save if requested
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(audio_bytes)
            
            # Play audio
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            play(audio)
            
        except Exception as e:
            print(f"✗ Speech error: {e}")
            print(f"   (Fallback to text): {text}")
    
    def listen(self, timeout: int = 10, phrase_time_limit: int = 15) -> Optional[str]:
        """
        Listen to user's voice input and convert to text using ElevenLabs.
        
        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for a single phrase
            
        Returns:
            Transcribed text or None if no speech detected
        """
        try:
            with self.microphone as source:
                print("🎧 Listening... (speak now)")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🔄 Processing speech...")
            
            # Get WAV audio data
            audio_data = audio.get_wav_data()
            
            # Use ElevenLabs speech-to-text
            try:
                result = self.elevenlabs_client.speech_to_text.convert(
                    audio=audio_data,
                    model_id="scribe-v1"  # ElevenLabs transcription model
                )
                
                text = result.text
                print(f"👤 You: {text}")
                return text
                
            except Exception as e:
                print(f"ElevenLabs transcription error: {e}")
                # Fallback to Google Speech Recognition (built into SpeechRecognition library)
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"👤 You (via Google): {text}")
                    return text
                except Exception as fallback_error:
                    print(f"Fallback transcription also failed: {fallback_error}")
                    return None
            
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout)")
            return None
        except Exception as e:
            print(f"✗ Listening error: {e}")
            return None
    
    # ==========================================
    # AI CONVERSATION ENGINE
    # ==========================================
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for the AI stylist."""
        closet_summary = self._summarize_closet()
        
        # Build profile insights section
        profile_insights = ""
        if self.user_context.get("inferred_style"):
            profile_insights = f"""
INFERRED USER PROFILE (from closet analysis):
- Style: {self.user_context.get('inferred_style')}
- Personality traits: {', '.join(self.user_context.get('personality_traits', []))}
- Favorite colors: {', '.join(self.user_context.get('color_preferences', []))}
- Formality preference: {self.user_context.get('formality_preference')}
- Lifestyle: {self.user_context.get('lifestyle')}
- Shopping style: {self.user_context.get('shopping_style')}
- Confidence level: {self.user_context.get('profile_confidence')}

IMPORTANT: Use this profile to inform your recommendations and tailor questions. 
- If you already know their style is minimalist, don't ask if they're bold or minimalist
- If you know their favorite colors, incorporate them naturally
- Skip questions where you already have strong insights from their closet
- Focus questions on the current occasion, not their general style
"""
        
        return f"""You are Bella, an expert AI personal stylist with 15 years of fashion consulting experience.
You have a warm, enthusiastic personality and genuinely care about making your clients look and feel their best.

CONVERSATION STYLE:
- Be conversational, friendly, and encouraging
- Ask ONE question at a time (never multiple questions in one response)
- Keep responses concise (2-3 sentences max)
- Show excitement about helping them look amazing
- Be empathetic to their concerns and preferences
- Use fashion terminology naturally but explain if needed
- Reference their existing style naturally ("I know you love minimalist looks...")

YOUR CLIENT'S CLOSET:
{closet_summary}

{profile_insights}

YOUR ADAPTIVE PROCESS:
1. Greet warmly and ask about the occasion/event
2. Understand the formality level (casual, business casual, formal, etc.)
3. Ask about the weather/season
4. Understand their current mood and how they want to feel
5. ONLY ask about personality/style if profile confidence is LOW
6. Ask if there are specific goals for this outfit (impress, comfort, confidence boost)
7. Once you have enough context, recommend a COMPLETE outfit with:
   - Main clothing item (dress/top/pants)
   - Shoes
   - Accessories (jewelry, bag, scarf, etc.)
   - Explain WHY this outfit works for their context AND their known style

IMPORTANT RULES:
- Only recommend items that exist in their closet (listed above)
- Use the inferred profile to make smarter recommendations
- Don't ask redundant questions - you already know their style from their closet
- If their closet lacks certain categories, mention it gently and suggest they could add it
- Consider emotional context: confidence boost, comfort, making an impression, etc.
- Match outfit to their known personality traits and style preferences
- Always provide the reasoning that connects to BOTH the occasion AND their personal style

Current conversation context: {json.dumps(self.user_context, indent=2)}
"""
    
    def _summarize_closet(self) -> str:
        """Create a summary of the user's closet for the AI."""
        if not self.user_closet:
            return "Empty closet (no items added yet)"
        
        # Group by category
        by_category = {}
        for item in self.user_closet:
            cat = item.get("category", "Unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        summary_lines = []
        for category, items in by_category.items():
            item_descriptions = []
            for item in items[:5]:  # Limit to first 5 per category
                desc = f"{item.get('color_hex', 'unknown color')} {item.get('material_inference', '')} {item.get('brand_guess', '')} (ID: {item['id']})"
                item_descriptions.append(desc)
            
            summary_lines.append(f"{category} ({len(items)} items): {', '.join(item_descriptions)}")
        
        return "\n".join(summary_lines)
    
    def chat(self, user_message: str) -> str:
        """
        Send message to Gemini 2.5 Flash and get stylist's response.
        
        Args:
            user_message: User's input text
            
        Returns:
            AI stylist's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages with system prompt
        system_prompt = self.get_system_prompt()
        
        # Convert conversation history to Gemini format
        gemini_messages = []
        
        # Add system instruction as first user message (Gemini doesn't have system role)
        gemini_messages.append({
            "role": "user",
            "parts": [{"text": f"SYSTEM INSTRUCTIONS:\n{system_prompt}"}]
        })
        gemini_messages.append({
            "role": "model",
            "parts": [{"text": "Understood. I'll follow these instructions as Bella, your personal stylist."}]
        })
        
        # Add conversation history
        for msg in self.conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        try:
            # Call Gemini 2.5 Flash
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=300
                )
            )
            
            assistant_message = response.text.strip()
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Extract context updates from the conversation
            self._update_context_from_conversation(user_message, assistant_message)
            
            return assistant_message
            
        except Exception as e:
            print(f"✗ Chat error: {e}")
            return "I apologize, I'm having trouble processing that. Could you repeat?"
    
    def _update_context_from_conversation(self, user_msg: str, assistant_msg: str) -> None:
        """Extract and update user context from conversation using GPT."""
        extraction_prompt = f"""Extract structured information from this conversation exchange:

User: {user_msg}
Assistant: {assistant_msg}

Extract any mentioned:
- event_type (e.g., "wedding", "job interview", "date", "casual outing", "dinner party")
- event_formality (casual/business casual/formal/black tie)
- weather (sunny/rainy/cold/hot/mild)
- mood (confident/nervous/excited/relaxed/anxious/happy)
- desired_feeling (how they WANT to feel: "confident", "comfortable", "powerful", "approachable", "sophisticated", "bold", "relaxed")
- outfit_goals (specific objectives: "make good impression", "feel comfortable all day", "stand out", "blend in", "express creativity")
- personality_traits (bold/minimalist/creative/classic/edgy/romantic)
- style_preferences (modern/vintage/bohemian/streetwear/preppy)
- color_preferences (specific colors they like or want to wear)
- body_concerns (specific areas they're conscious about or want to highlight/minimize)
- occasion_details (any other relevant details about the event, time, location, attendees)

Return ONLY a JSON object with the extracted fields (omit fields if not mentioned):
"""
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part(text=extraction_prompt)],
                config=types.GenerateContentConfig(
                    temperature=0
                )
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to parse JSON (handle markdown code blocks if present)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
            else:
                extracted = json.loads(response_text)
            
            # Update context (merge with existing)
            for key, value in extracted.items():
                if key in self.user_context:
                    if isinstance(self.user_context[key], list) and value:
                        # Append to lists
                        if isinstance(value, list):
                            self.user_context[key].extend(value)
                        else:
                            self.user_context[key].append(value)
                    elif value:
                        # Replace single values
                        self.user_context[key] = value
                        
        except Exception as e:
            # Silent fail - context extraction is nice to have but not critical
            pass
    
    # ==========================================
    # OUTFIT RECOMMENDATION
    # ==========================================
    
    def generate_outfit_recommendation(self) -> Dict:
        """
        Generate final outfit recommendation based on comprehensive context analysis.
        
        Considers:
        1. User's inferred style from closet analysis
        2. Personality traits (practical, bold, creative, etc.)
        3. Previous scene contexts from VLM (where they go, what they do)
        4. Current event type and formality
        5. Weather/season
        6. Current mood and desired feeling
        7. Color preferences and patterns
        
        Returns:
            Dictionary with outfit items and reasoning
        """
        # SCENARIO 5: Empty closet check
        if not self.user_closet or len(self.user_closet) == 0:
            return {
                "error": "empty_closet",
                "message": "I notice your closet is empty! To get personalized outfit recommendations, please upload some photos of your clothing items first. I'll analyze them and create amazing outfits for you!",
                "action_required": "upload_items"
            }
        
        # SCENARIO 6: Single category check
        categories = {}
        for item in self.user_closet:
            cat = item.get('category', 'unknown').lower()
            categories[cat] = categories.get(cat, 0) + 1
        
        required_categories = ['top', 'bottom', 'shoes', 'shirt', 'pants', 'dress', 'skirt', 'jacket', 'blazer']
        has_top = any(cat in categories for cat in ['top', 'shirt', 't-shirt', 'blouse', 'sweater', 'jacket', 'blazer'])
        has_bottom = any(cat in categories for cat in ['bottom', 'pants', 'jeans', 'skirt', 'shorts', 'dress'])
        has_shoes = any(cat in categories for cat in ['shoes', 'sneakers', 'heels', 'boots', 'sandals'])
        
        missing_categories = []
        if not has_top:
            missing_categories.append("tops (shirts, blouses, jackets)")
        if not has_bottom:
            missing_categories.append("bottoms (pants, skirts, dresses)")
        if not has_shoes:
            missing_categories.append("shoes")
        
        if missing_categories:
            return {
                "error": "incomplete_wardrobe",
                "message": f"I see you have {len(self.user_closet)} items, but your closet is missing some essential categories: {', '.join(missing_categories)}. For complete outfit recommendations, try uploading items in these categories!",
                "current_items": len(self.user_closet),
                "missing_categories": missing_categories,
                "action_required": "upload_more_items"
            }
        
        self._thinking("Personalizing an outfit just for you.")
        
        # SCENARIO 16: Resolve VLM/closet conflicts
        self.resolve_vlm_closet_conflict()
        
        # SCENARIO 7: Remove duplicate items before recommendation
        unique_closet = self.check_for_duplicates(self.user_closet)
        
        # SCENARIO 10: Check for multi-event day
        multi_event_strategy = None
        if self.user_context.get('is_multi_event') and len(self.user_context.get('events', [])) > 1:
            multi_event_strategy = self.handle_multi_event_day(self.user_context['events'])
            if multi_event_strategy and multi_event_strategy['strategy'] == 'outfit_change':
                # Need separate outfits - return early
                return {
                    "multi_event": True,
                    "strategy": "outfit_change",
                    "message": multi_event_strategy['message'],
                    "transition_tip": multi_event_strategy['transition_tip'],
                    "outfit_1_context": self.user_context['events'][0],
                    "outfit_2_context": self.user_context['events'][1]
                }
        
        # Build comprehensive recommendation context
        recommendation_prompt = f"""You are creating a personalized outfit recommendation. Consider ALL of these factors:

═══════════════════════════════════════════════════════════════════
1. USER'S ESTABLISHED STYLE & PERSONALITY
═══════════════════════════════════════════════════════════════════
Inferred Style: {self.user_context.get('inferred_style', 'Unknown')}
Personality Traits: {', '.join(self.user_context.get('personality_traits', []))}
Favorite Colors: {', '.join(self.user_context.get('color_preferences', []))}
Formality Preference: {self.user_context.get('formality_preference', 'Unknown')}
Lifestyle: {self.user_context.get('lifestyle', 'Unknown')}
Shopping Style: {self.user_context.get('shopping_style', 'Unknown')}

═══════════════════════════════════════════════════════════════════
2. PREVIOUS SCENE CONTEXTS (What they typically do/wear)
═══════════════════════════════════════════════════════════════════
Past VLM Analysis: {json.dumps(self.user_context.get('vlm_history', []), indent=2)}
- Look for patterns in where they go (parks, offices, cafes, etc.)
- What activities they do (casual outings, professional settings, social events)
- Their comfort zones and typical scenarios

═══════════════════════════════════════════════════════════════════
3. CURRENT EVENT CONTEXT
═══════════════════════════════════════════════════════════════════
Event Type: {self.user_context.get('event_type', 'Unknown')}
Formality Level: {self.user_context.get('event_formality', 'Unknown')}
Weather/Season: {self.user_context.get('weather', 'Unknown')}
Occasion Details: {self.user_context.get('occasion_details', 'None provided')}

═══════════════════════════════════════════════════════════════════
4. EMOTIONAL & PSYCHOLOGICAL NEEDS
═══════════════════════════════════════════════════════════════════
Current Mood: {self.user_context.get('mood', 'Unknown')}
Desired Feeling: {self.user_context.get('desired_feeling', 'Unknown')}
Body Concerns: {', '.join(self.user_context.get('body_concerns', []))}
Specific Goals: {self.user_context.get('outfit_goals', 'None stated')}

═══════════════════════════════════════════════════════════════════
5. AVAILABLE CLOSET ITEMS
═══════════════════════════════════════════════════════════════════
{self._summarize_closet()}

═══════════════════════════════════════════════════════════════════
RECOMMENDATION STRATEGY
═══════════════════════════════════════════════════════════════════

Your task is to select items that:

✓ ALIGN WITH THEIR CORE STYLE
  - Match their inferred style ({self.user_context.get('inferred_style')})
  - Reflect their personality traits ({', '.join(self.user_context.get('personality_traits', []))})
  - Use their preferred colors when possible

✓ FIT THE OCCASION
  - Appropriate formality for {self.user_context.get('event_type')}
  - Suitable for {self.user_context.get('weather')} weather
  - Matches the event's vibe and context

✓ ADDRESS EMOTIONAL NEEDS
  - Help them feel {self.user_context.get('desired_feeling', 'confident and comfortable')}
  - Support their current mood: {self.user_context.get('mood', 'neutral')}
  - Provide confidence boost through styling

✓ REFERENCE THEIR HISTORY
  - Consider their typical scene contexts (past VLM data)
  - Build on what has worked for them before
  - Stay within their comfort zone or slightly stretch it based on event

✓ PRACTICAL CONSIDERATIONS
  - Weather-appropriate layering
  - Comfortable for the activities involved
  - Address any stated body concerns

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════

Provide a JSON response with:
{{
    "outfit": {{
        "top": {{
            "id": <item_id from closet>,
            "reasoning": "Why this works: [style fit] + [occasion fit] + [emotional support]"
        }},
        "bottom": {{
            "id": <item_id>,
            "reasoning": "Complete explanation considering all factors"
        }},
        "shoes": {{
            "id": <item_id>,
            "reasoning": "How this complements the outfit and occasion"
        }},
        "accessories": [
            {{"id": <item_id>, "reasoning": "Finishing touch that elevates the look"}}
        ],
        "outerwear": {{
            "id": <item_id>,
            "reasoning": "Weather protection + style enhancement"
        }} (only if weather requires it)
    }},
    "overall_vibe": "Describe how this complete outfit creates the desired feeling and matches their style",
    "confidence_boost": "Specific ways this outfit will make them feel {self.user_context.get('desired_feeling', 'amazing')} and address their mood",
    "styling_tips": [
        "Tip 1: How to wear/style this outfit",
        "Tip 2: What to pay attention to",
        "Tip 3: How this builds on their usual style"
    ],
    "reasoning_summary": {{
        "style_alignment": "How outfit matches their {self.user_context.get('inferred_style')} style",
        "personality_fit": "How it reflects their {', '.join(self.user_context.get('personality_traits', []))} traits",
        "occasion_appropriateness": "Why perfect for {self.user_context.get('event_type')}",
        "emotional_support": "How it delivers {self.user_context.get('desired_feeling')} feeling"
    }}
}}

CRITICAL RULES:
1. Only use item IDs that actually exist in the closet above
2. If a category is missing from closet, set it to null and mention in reasoning
3. Every reasoning must reference MULTIPLE factors (style + occasion + emotion)
4. Connect recommendations back to their known preferences and history
5. Be specific about HOW the outfit achieves the desired feeling
"""
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part(text=recommendation_prompt)],
                config=types.GenerateContentConfig(
                    temperature=0.7
                )
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to parse JSON (handle markdown code blocks if present)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                recommendation = json.loads(json_match.group())
            else:
                recommendation = json.loads(response_text)
            
            # SCENARIO 18: Check for repetition
            if self.avoid_repetition(recommendation):
                print("⚠️  Outfit too similar to recent recommendation, adjusting...")
                # Try to get a different combination (simple: swap one item)
                # In production, might re-call with constraint to avoid specific items
            
            # SCENARIO 20: Validate color coordination
            color_validation = self.validate_color_coordination(recommendation.get('outfit', {}))
            
            if not color_validation['is_valid']:
                print(f"⚠️  Color coordination issues detected:")
                for issue in color_validation['issues']:
                    print(f"    - {issue}")
                
                # Add warnings to recommendation
                recommendation['color_warnings'] = color_validation['issues']
                recommendation['color_suggestions'] = color_validation['suggestions']
            
            # SCENARIO 18: Save to history
            self.save_recommendation(recommendation)
            
            return recommendation
            
        except Exception as e:
            print(f"✗ Recommendation error: {e}")
            return {
                "outfit": {},
                "overall_vibe": "I need more information to make a recommendation.",
                "confidence_boost": "",
                "styling_tips": []
            }
    
    def format_outfit_presentation(self, recommendation: Dict) -> str:
        """Format the outfit recommendation into a conversational presentation."""
        outfit = recommendation.get("outfit", {})
        vibe = recommendation.get("overall_vibe", "")
        confidence = recommendation.get("confidence_boost", "")
        tips = recommendation.get("styling_tips", [])
        
        # Build presentation
        presentation = f"Perfect! I've put together an amazing outfit for you. {vibe}\n\n"
        
        # List each item
        for category, item_data in outfit.items():
            if item_data and isinstance(item_data, dict):
                item_id = item_data.get("id")
                reasoning = item_data.get("reasoning", "")
                
                # Find the actual item
                closet_item = next((i for i in self.user_closet if i["id"] == item_id), None)
                if closet_item:
                    presentation += f"{category.title()}: {closet_item.get('color_hex')} {closet_item.get('material_inference')} - {reasoning}\n"
        
        presentation += f"\n{confidence}\n"
        
        if tips:
            presentation += "\nStyling tips:\n"
            for tip in tips:
                presentation += f"• {tip}\n"
        
        return presentation
    
    def create_outfit_collage(self, recommendation: Dict) -> Optional[str]:
        """
        Create Instagram-worthy outfit collage and upload to Supabase.
        
        Args:
            recommendation: Outfit recommendation with item IDs
            
        Returns:
            Public URL of the collage image, or None if failed
        """
        try:
            from PIL import Image, ImageDraw, ImageFilter
            import requests
            from io import BytesIO
            
            outfit = recommendation.get("outfit", {})
            
            # Collect item images
            item_images = {}
            
            for category in ["top", "bottom", "shoes", "accessories", "bag", "outerwear"]:
                item_data = outfit.get(category)
                
                if not item_data:
                    continue
                
                if category == "accessories" and isinstance(item_data, list):
                    # Handle multiple accessories
                    item_images["accessories"] = []
                    for acc in item_data[:2]:  # Max 2 accessories
                        item_id = acc.get("id")
                        closet_item = next((i for i in self.user_closet if i["id"] == item_id), None)
                        if closet_item and closet_item.get("image_path"):
                            item_images["accessories"].append({
                                "url": closet_item["image_path"],
                                "name": closet_item.get("category", "Accessory")
                            })
                else:
                    # Handle single items
                    item_id = item_data.get("id")
                    closet_item = next((i for i in self.user_closet if i["id"] == item_id), None)
                    
                    if closet_item and closet_item.get("image_path"):
                        # Map to collage categories
                        if category in ["top", "outerwear"]:
                            item_images["topwear"] = closet_item["image_path"]
                        elif category == "bottom":
                            item_images["bottomwear"] = closet_item["image_path"]
                        elif category == "shoes":
                            item_images["shoes"] = closet_item["image_path"]
                        elif category == "bag":
                            item_images["bag"] = closet_item["image_path"]
            
            if not item_images:
                print("No images to create collage")
                return None
            
            # SCENARIO 14: Validate all images before creating collage
            validated_items = []
            for category, url_data in item_images.items():
                if category == "accessories" and isinstance(url_data, list):
                    valid_accessories = []
                    for acc in url_data:
                        if self._is_valid_image_url(acc["url"]):
                            valid_accessories.append(acc)
                        else:
                            print(f"⚠️  Skipping accessory with invalid image: {acc.get('name')}")
                    if valid_accessories:
                        validated_items.append((category, valid_accessories))
                else:
                    if self._is_valid_image_url(url_data):
                        validated_items.append((category, url_data))
                    else:
                        print(f"⚠️  Skipping {category} with invalid image URL")
            
            if not validated_items:
                print("✗ No valid images found for collage")
                return None
            
            # Convert back to dict
            item_images = {}
            for category, url_data in validated_items:
                item_images[category] = url_data
            
            # Create collage
            canvas_width = 800
            canvas_height = 1200
            padding = 40
            
            # Create white canvas
            canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            
            # Add elegant border
            border_width = 3
            border_color = (224, 224, 224, 255)  # Light gray
            draw.rectangle(
                [padding//2, padding//2, canvas_width - padding//2, canvas_height - padding//2],
                outline=border_color,
                width=border_width
            )
            
            # Layout positions
            layout = {
                "topwear": {"pos": (400, 150), "size": (300, 350)},
                "bottomwear": {"pos": (400, 550), "size": (300, 400)},
                "shoes": {"pos": (400, 1000), "size": (280, 250)},
                "bag": {"pos": (650, 300), "size": (180, 180)},
                "accessories": [
                    {"pos": (150, 300), "size": (140, 140)},
                    {"pos": (150, 700), "size": (140, 140)}
                ]
            }
            
            def download_and_place_item(url, position, size):
                """Download image from URL and place on canvas with shadow."""
                try:
                    # Handle base64 data URLs
                    if url.startswith('data:image'):
                        # Extract base64 data
                        base64_data = url.split(',')[1]
                        img_data = base64.b64decode(base64_data)
                        item_img = Image.open(BytesIO(img_data)).convert('RGBA')
                    else:
                        # Download from URL
                        response = requests.get(url, timeout=5)
                        item_img = Image.open(BytesIO(response.content)).convert('RGBA')
                    
                    # Resize maintaining aspect ratio
                    item_img.thumbnail(size, Image.Resampling.LANCZOS)
                    
                    # Create shadow
                    shadow = Image.new('RGBA', (item_img.width + 20, item_img.height + 20), (0, 0, 0, 0))
                    shadow_draw = ImageDraw.Draw(shadow)
                    shadow_draw.ellipse([0, 0, shadow.width, shadow.height], fill=(0, 0, 0, 40))
                    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
                    
                    # Calculate centered position
                    x = position[0] - item_img.width // 2
                    y = position[1] - item_img.height // 2
                    
                    # Paste shadow
                    canvas.paste(shadow, (x - 10, y - 10), shadow)
                    
                    # Paste item
                    canvas.paste(item_img, (x, y), item_img)
                    
                    return True
                except Exception as e:
                    print(f"Error placing item from {url}: {e}")
                    return False
            
            # Place items
            for category, url in item_images.items():
                if category == "accessories":
                    # Handle multiple accessories
                    for i, acc in enumerate(url):
                        if i < len(layout["accessories"]):
                            download_and_place_item(
                                acc["url"],
                                layout["accessories"][i]["pos"],
                                layout["accessories"][i]["size"]
                            )
                elif category in layout:
                    download_and_place_item(
                        url,
                        layout[category]["pos"],
                        layout[category]["size"]
                    )
            
            # Convert to RGB for JPEG
            final_canvas = Image.new('RGB', canvas.size, (255, 255, 255))
            final_canvas.paste(canvas, (0, 0), canvas)
            
            # Save to bytes
            buf = BytesIO()
            final_canvas.save(buf, format='PNG', quality=95)
            buf.seek(0)
            
            # Upload to Supabase Storage
            collage_url = self._upload_collage_to_storage(buf.getvalue())
            
            return collage_url
            
        except Exception as e:
            print(f"Error creating collage: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _upload_collage_to_storage(self, image_bytes: bytes) -> Optional[str]:
        """Upload collage to storage via API and return public URL."""
        try:
            import requests
            from datetime import datetime
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"outfit_{timestamp}.png"
            
            # Encode to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Upload via API
            response = requests.post(
                f"{self.closet_api_url.replace('/api', '')}/api/upload-collage",
                json={
                    "image_base64": image_b64,
                    "filename": filename
                },
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    url = data.get('url')
                    print(f"✅ Collage uploaded: {url}")
                    return url
            
            print(f"Upload failed: {response.status_code}")
            return None
            
        except Exception as e:
            print(f"Error uploading collage: {e}")
            return None
    
    def _is_valid_image_url(self, url: str) -> bool:
        """
        SCENARIO 14: Validate image URL.
        
        Args:
            url: Image URL or data URL
            
        Returns:
            True if URL appears valid
        """
        if not url:
            return False
        
        # Valid formats
        if url.startswith('data:image'):
            return True
        
        if url.startswith('http://') or url.startswith('https://'):
            # Could add actual URL validation/ping here
            return True
        
        if url.startswith('/'):
            # Relative path - assume valid
            return True
        
        return False
    
    # ==========================================
    # MAIN CONVERSATION FLOW
    # ==========================================
    
    def start_styling_session(self) -> None:
        """Main conversation loop for the styling session."""
        print("\n" + "="*60)
        print("  AI PERSONAL STYLIST - Voice Session")
        print("="*60 + "\n")
        
        # Fetch closet
        print("Loading your closet...")
        self.fetch_user_closet()
        
        if not self.user_closet:
            self.speak("Hi! I notice your closet is empty. Please add some items to your digital closet first, and then I can help you create amazing outfits!")
            return
        
        # Start conversation with personalized greeting based on profile
        if self.user_context.get("inferred_style"):
            style = self.user_context.get("inferred_style", "unique")
            greeting = f"Hi! I'm Bella, your personal stylist! I've been looking at your closet and I love your {style} style! I'm so excited to help you look absolutely amazing today. What's the occasion?"
        else:
            greeting = "Hi! I'm Bella, your personal stylist! I'm so excited to help you look absolutely amazing today. What's the occasion?"
        
        self.speak(greeting)
        
        # Conversation loop
        turns = 0
        max_turns = 10
        
        while turns < max_turns:
            # Listen to user
            user_input = self.listen()
            
            if user_input is None:
                self.speak("I didn't catch that. Could you say it again?")
                continue
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ["exit", "quit", "bye", "goodbye"]):
                self.speak("It was wonderful styling you today! Come back anytime you need outfit advice. Bye!")
                break
            
            # Get AI response
            response = self.chat(user_input)
            self.speak(response)
            
            turns += 1
            
            # After gathering context, check if ready to recommend
            if turns >= 5 and self.user_context.get("event_type"):
                # Ask if ready for recommendation
                if "recommend" in user_input.lower() or "outfit" in user_input.lower():
                    print("\n🎨 Generating your perfect outfit...\n")
                    recommendation = self.generate_outfit_recommendation()
                    presentation = self.format_outfit_presentation(recommendation)
                    self.speak(presentation)
                    
                    # Create visual collage
                    self.speak("Let me create a visual collage of this outfit for you...")
                    collage_url = self.create_outfit_collage(recommendation)
                    
                    if collage_url:
                        self.speak(f"Perfect! Here's your complete outfit laid out. You can view it at: {collage_url}")
                        print(f"\n🖼️  Outfit Collage: {collage_url}\n")
                    else:
                        self.speak("I had a small technical issue creating the visual, but the outfit I described will look amazing on you!")
                    
                    # Ask for feedback
                    self.speak("What do you think? Would you like me to make any changes?")
                    
                    follow_up = self.listen()
                    if follow_up and any(word in follow_up.lower() for word in ["show", "see", "items"]):
                        self._display_outfit_items(recommendation)
                    elif follow_up and any(word in follow_up.lower() for word in ["change", "different", "other"]):
                        change_response = self.chat(f"User wants changes: {follow_up}")
                        self.speak(change_response)
                    
                    break
        
        # Session complete
        print("\n" + "="*60)
        print("  Styling Session Complete")
        print("="*60 + "\n")
    
    def _display_outfit_items(self, recommendation: Dict) -> None:
        """Display the recommended outfit items with image paths."""
        outfit = recommendation.get("outfit", {})
        
        print("\n📸 YOUR OUTFIT ITEMS:\n")
        for category, item_data in outfit.items():
            if item_data and isinstance(item_data, dict):
                item_id = item_data.get("id")
                closet_item = next((i for i in self.user_closet if i["id"] == item_id), None)
                
                if closet_item:
                    print(f"{category.upper()}:")
                    print(f"  • Image: {self.closet_api_url}{closet_item.get('image_path')}")
                    print(f"  • Details: {closet_item.get('color_hex')} {closet_item.get('material_inference')}")
                    print(f"  • Why: {item_data.get('reasoning')}\n")
    
    # ============================================================================
    # SCENARIO ENHANCEMENTS - ALL 4 SCENARIOS
    # ============================================================================
    
    def calculate_item_relevancy(self, item: Dict, context: Dict) -> float:
        """
        SCENARIO 2: Calculate 0-100% relevancy score for an item given the context.
        
        Used to determine if closet items match the occasion.
        
        Scoring breakdown:
        - Occasion Match (40%): Does item vibe match event type?
        - Style Alignment (30%): Does item style match user's inferred style?
        - Color Preference (20%): Is item color in user's favorites?
        - Season Appropriateness (10%): Does item fit the weather?
        
        Returns:
            float: Relevancy score 0-100%
        """
        score = 0.0
        
        # 1. OCCASION MATCH (40%)
        event_type = context.get("event_type", "").lower()
        item_vibe = item.get("occasion_vibe", "").lower()
        
        # Event-specific scoring matrix
        occasion_scores = {
            "business meeting": {
                "formal": 40, "business casual": 35, "business": 40, 
                "professional": 38, "smart casual": 30
            },
            "interview": {
                "business": 40, "formal": 38, "professional": 40, 
                "business casual": 35, "smart casual": 25
            },
            "wedding": {
                "formal": 40, "elegant": 38, "dressy": 35, 
                "party": 30, "evening": 35
            },
            "party": {
                "party": 40, "evening": 38, "social": 35, 
                "dressy": 33, "casual": 25, "smart casual": 30
            },
            "date": {
                "dressy": 38, "evening": 35, "smart casual": 36,
                "casual": 30, "romantic": 40
            },
            "cafe": {
                "casual": 38, "smart casual": 40, "relaxed": 35, 
                "everyday": 33, "comfortable": 35
            },
            "gym": {
                "athletic": 40, "sporty": 38, "workout": 40, 
                "active": 35, "performance": 38
            },
            "beach": {
                "beach": 40, "resort": 38, "summer": 35, 
                "casual": 30, "relaxed": 33
            }
        }
        
        # Match event type to item vibe
        if event_type in occasion_scores:
            for vibe_keyword, points in occasion_scores[event_type].items():
                if vibe_keyword in item_vibe:
                    score += points
                    break
            else:
                # Partial match for "casual" in most non-formal events
                if "casual" in item_vibe and event_type not in ["business meeting", "interview", "wedding"]:
                    score += 20
        else:
            # Generic event - casual works moderately
            if "casual" in item_vibe:
                score += 25
        
        # 2. STYLE ALIGNMENT (30%)
        inferred_style = self.user_context.get("inferred_style", "").lower()
        item_style = str(item.get("style", "")).lower()
        
        if inferred_style and item_style:
            # Exact or very close match
            if inferred_style in item_style or item_style in inferred_style:
                score += 30
            # Partial match (any word matches)
            elif any(word in item_style for word in inferred_style.split() if len(word) > 3):
                score += 20
            # Complementary styles
            elif (("minimalist" in inferred_style and any(x in item_style for x in ["modern", "clean", "simple"])) or
                  ("bohemian" in inferred_style and any(x in item_style for x in ["relaxed", "free", "boho"])) or
                  ("classic" in inferred_style and any(x in item_style for x in ["timeless", "traditional"]))):
                score += 25
        
        # 3. COLOR PREFERENCE (20%)
        color_prefs = self.user_context.get("color_preferences", [])
        item_color = item.get("color_hex", "")
        
        if color_prefs and item_color:
            # Exact color match
            if item_color in color_prefs:
                score += 20
            # Color family match (similar brightness/tone)
            elif any(self._colors_similar(item_color, pref) for pref in color_prefs):
                score += 15
            # Neutral colors (always somewhat relevant)
            elif item_color in ["#FFFFFF", "#000000", "#808080", "#F5F5DC", "#C0C0C0"]:
                score += 10
        
        # 4. SEASON APPROPRIATENESS (10%)
        weather = context.get("weather", "").lower()
        item_season = str(item.get("seasonality", "")).lower()
        
        season_keywords = {
            "hot": ["summer", "spring"],
            "warm": ["summer", "spring"],
            "sunny": ["summer", "spring"],
            "cold": ["winter", "fall"],
            "cool": ["fall", "spring"],
            "chilly": ["fall", "winter"],
            "rainy": ["fall", "spring"],
            "snowy": ["winter"]
        }
        
        for weather_key, appropriate_seasons in season_keywords.items():
            if weather_key in weather:
                if any(season in item_season for season in appropriate_seasons):
                    score += 10
                elif "all season" in item_season or not item_season:
                    score += 5
                break
        else:
            # No specific weather mentioned
            if item_season:
                score += 5
        
        return min(score, 100.0)
    
    def _colors_similar(self, color1: str, color2: str) -> bool:
        """Check if two hex colors are in the same family (simplified)."""
        try:
            if not color1.startswith('#') or not color2.startswith('#'):
                return False
            
            # Convert hex to RGB
            r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
            r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
            
            # Calculate brightness
            bright1 = (r1 + g1 + b1) / 3
            bright2 = (r2 + g2 + b2) / 3
            
            # Colors are similar if brightness difference < 100
            return abs(bright1 - bright2) < 100
        except:
            return False
    
    def analyze_inspiration_image(self, image_url: str) -> Dict:
        """
        SCENARIO 3: Analyze outfit inspiration image using Mistral VLM.
        
        Takes an inspiration image URL and extracts style details.
        
        Args:
            image_url: URL, file path, or base64 data URL of inspiration image
            
        Returns:
            dict: {
                "items": ["specific items seen"],
                "style": "overall style aesthetic",
                "colors": ["dominant colors"],
                "vibe": "mood/vibe description",
                "key_features": ["notable elements"]
            }
        """
        try:
            from mistralai import Mistral
            import base64
            
            # Get Mistral client
            mistral_key = os.getenv("MISTRAL_API_KEY")
            if not mistral_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")
            
            client = Mistral(api_key=mistral_key)
            
            # Prepare image data (handle different input formats)
            if image_url.startswith('data:image'):
                # Already base64 data URL
                image_data = image_url
            elif image_url.startswith('http'):
                # Download and convert to base64
                response = requests.get(image_url, timeout=10)
                b64 = base64.b64encode(response.content).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{b64}"
            else:
                # Assume local file path
                with open(image_url, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{b64}"
            
            # Analyze with Mistral VLM
            prompt = """Analyze this outfit inspiration image in detail.
Return ONLY a valid JSON object with these exact fields:
{
  "items": ["list specific clothing items you see, e.g., white blouse, high-waist jeans, sandals"],
  "style": "overall style aesthetic in 2-3 words, e.g., effortless casual, polished minimalist",
  "colors": ["list dominant colors in the outfit, e.g., white, blue, beige"],
  "vibe": "describe the mood/vibe in 2-3 words, e.g., relaxed summery, edgy urban",
  "key_features": ["notable design elements or styling, e.g., flowy fabric, high waist, neutral tones"]
}
Return ONLY valid JSON. No markdown code blocks, no explanations."""
            
            response = client.chat.complete(
                model=os.getenv("MISTRAL_MODEL", "pixtral-large-latest"),
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            
            # Parse JSON response
            import re
            text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\n?", "", text)
            
            # Extract JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return json.loads(text)
            
        except Exception as e:
            print(f"Error analyzing inspiration image: {e}")
            return {
                "items": [],
                "style": "unknown",
                "colors": [],
                "vibe": "casual",
                "key_features": []
            }
    
    def find_similar_items(self, inspo_analysis: Dict, threshold: float = 65.0) -> List[Dict]:
        """
        SCENARIO 3: Find closet items similar to inspiration.
        
        Searches closet for items matching the inspiration style/colors/vibe.
        
        Args:
            inspo_analysis: Output from analyze_inspiration_image()
            threshold: Minimum similarity score (0-100) to include
            
        Returns:
            List of matches sorted by similarity score, each containing:
            - item: The closet item dict
            - score: Similarity score 0-100
            - reasoning: Explanation of why it matches
            
            OR dict with "found_matches": False if no items found (SCENARIO 12)
        """
        similar = []
        
        for item in self.user_closet:
            score = self._calculate_inspiration_similarity(inspo_analysis, item)
            
            if score >= threshold:
                similar.append({
                    "item": item,
                    "score": round(score, 1),
                    "reasoning": self._explain_similarity(inspo_analysis, item, score)
                })
        
        sorted_similar = sorted(similar, key=lambda x: x['score'], reverse=True)
        
        # SCENARIO 12: Handle no similar items found
        if not sorted_similar:
            return self.handle_no_similar_items(inspo_analysis)
        
        return sorted_similar
    
    def _calculate_inspiration_similarity(self, inspo: Dict, item: Dict) -> float:
        """Calculate 0-100% similarity between inspiration and closet item."""
        score = 0.0
        
        # 1. STYLE MATCH (40%)
        inspo_style = inspo.get('style', '').lower()
        item_style = str(item.get('style', '')).lower()
        
        if inspo_style and item_style:
            # Exact or substring match
            if inspo_style in item_style or item_style in inspo_style:
                score += 40
            # Partial word match
            elif any(word in item_style for word in inspo_style.split() if len(word) > 3):
                score += 25
            # Generic style match
            elif any(x in item_style for x in ["casual", "modern", "classic"]):
                score += 15
        
        # 2. COLOR SIMILARITY (30%)
        inspo_colors = [c.lower() for c in inspo.get('colors', [])]
        item_color_hex = str(item.get('color_hex', ''))
        
        # Try to match color names (simplified)
        color_name_map = {
            "#FFFFFF": "white", "#000000": "black", "#808080": "gray",
            "#FF0000": "red", "#0000FF": "blue", "#00FF00": "green",
            "#FFFF00": "yellow", "#FFA500": "orange", "#800080": "purple",
            "#FFC0CB": "pink", "#A52A2A": "brown", "#F5F5DC": "beige",
            "#1560BD": "blue", "#1E3A8A": "navy", "#ADD8E6": "light blue"
        }
        
        item_color_name = color_name_map.get(item_color_hex, "").lower()
        
        color_matched = False
        for inspo_color in inspo_colors:
            # Direct name match
            if inspo_color in item_color_name or item_color_name in inspo_color:
                score += 30
                color_matched = True
                break
            # Check for color families
            if (("blue" in inspo_color and "blue" in item_color_name) or
                ("black" in inspo_color and item_color_hex == "#000000") or
                ("white" in inspo_color and item_color_hex == "#FFFFFF")):
                score += 25
                color_matched = True
                break
        
        # Neutral colors get partial credit
        if not color_matched and item_color_hex in ["#FFFFFF", "#000000", "#808080", "#F5F5DC"]:
            score += 15
        
        # 3. VIBE MATCH (20%)
        inspo_vibe = inspo.get('vibe', '').lower()
        item_vibe = str(item.get('occasion_vibe', '')).lower()
        
        if inspo_vibe and item_vibe:
            # Exact or substring match
            if inspo_vibe in item_vibe or item_vibe in inspo_vibe:
                score += 20
            # Partial word match
            elif any(word in item_vibe for word in inspo_vibe.split() if len(word) > 4):
                score += 12
            # Generic vibe compatibility
            elif ("casual" in inspo_vibe and "casual" in item_vibe):
                score += 15
        
        # 4. CATEGORY/ITEM MATCH (10%)
        inspo_items = [it.lower() for it in inspo.get('items', [])]
        item_category = str(item.get('category', '')).lower()
        
        for inspo_item in inspo_items:
            # Check if category mentioned in inspiration items
            if any(cat_word in inspo_item for cat_word in item_category.split()):
                score += 10
                break
            # Check if inspiration item mentioned in category
            if any(inspo_word in item_category for inspo_word in inspo_item.split()):
                score += 10
                break
        
        return min(score, 100.0)
    
    def _explain_similarity(self, inspo: Dict, item: Dict, score: float) -> str:
        """Generate human-readable explanation of similarity."""
        reasons = []
        
        # Score-based description
        if score >= 90:
            reasons.append("Almost identical match")
        elif score >= 80:
            reasons.append("Very similar")
        elif score >= 70:
            reasons.append("Great match")
        else:
            reasons.append("Good match")
        
        # Specific reasons
        inspo_style = inspo.get('style', '').lower()
        item_style = str(item.get('style', '')).lower()
        if inspo_style and inspo_style in item_style:
            reasons.append(f"matches {inspo_style} style")
        
        inspo_colors = inspo.get('colors', [])
        if inspo_colors:
            reasons.append("similar color palette")
        
        inspo_vibe = inspo.get('vibe', '').lower()
        item_vibe = str(item.get('occasion_vibe', '')).lower()
        if inspo_vibe and inspo_vibe in item_vibe:
            reasons.append(f"same {inspo_vibe} vibe")
        
        return " - " + ", ".join(reasons)
    
    def detect_wardrobe_gaps(self, context: Dict, avg_relevancy: float) -> Dict:
        """
        SCENARIO 4: Identify wardrobe gaps when relevancy is low (<70%).
        
        Analyzes what's missing from the closet for the given occasion.
        
        Args:
            context: Event context dict with event_type, weather, etc.
            avg_relevancy: Average relevancy score of current closet
            
        Returns:
            dict: {
                "has_gaps": True,
                "avg_relevancy": float,
                "gaps": [list of missing items],
                "ideal_outfit": "description of ideal outfit",
                "message": "voice-friendly gap explanation"
            }
        """
        gaps = []
        
        # Event-specific requirements
        event_requirements = {
            "business meeting": {
                "required": ["blazer", "dress shirt", "dress pants", "dress shoes"],
                "nice_to_have": ["tie", "belt", "briefcase", "watch"]
            },
            "interview": {
                "required": ["blazer or professional top", "dress pants or skirt", "dress shoes"],
                "nice_to_have": ["portfolio", "conservative accessories", "belt"]
            },
            "wedding": {
                "required": ["formal dress or suit", "heels or dress shoes", "clutch or small bag"],
                "nice_to_have": ["jewelry", "formal outerwear", "dressy shawl"]
            },
            "party": {
                "required": ["dressy top or dress", "nice bottoms or skirt", "heels or dressy shoes"],
                "nice_to_have": ["statement jewelry", "clutch", "bold accessories"]
            },
            "gym": {
                "required": ["athletic top", "athletic bottoms", "sneakers"],
                "nice_to_have": ["sports bra", "headband", "gym bag", "water bottle"]
            },
            "date": {
                "required": ["dressy top or dress", "nice bottoms", "heels or nice shoes"],
                "nice_to_have": ["jewelry", "small bag", "light jacket"]
            },
            "cafe": {
                "required": ["casual top", "jeans or casual pants", "casual shoes"],
                "nice_to_have": ["light jacket", "bag", "sunglasses"]
            },
            "beach": {
                "required": ["swimwear", "cover-up", "sandals"],
                "nice_to_have": ["sun hat", "sunglasses", "beach bag"]
            }
        }
        
        event_type = context.get("event_type", "").lower()
        requirements = event_requirements.get(event_type, {"required": [], "nice_to_have": []})
        
        # Check for missing REQUIRED items
        for req_item in requirements["required"]:
            has_item = any(
                any(keyword in item.get('category', '').lower() 
                    for keyword in req_item.lower().split())
                for item in self.user_closet
            )
            
            if not has_item:
                gaps.append({
                    "category": req_item,
                    "priority": "essential",
                    "reasoning": f"Essential for {event_type}",
                    "style_suggestion": self._suggest_style_for_item(req_item)
                })
        
        # Check for NICE-TO-HAVE items (only if few essential gaps)
        if len(gaps) < 2:
            for nice_item in requirements["nice_to_have"][:3]:  # Limit to 3
                has_item = any(
                    any(keyword in item.get('category', '').lower()
                        for keyword in nice_item.lower().split())
                    for item in self.user_closet
                )
                
                if not has_item:
                    gaps.append({
                        "category": nice_item,
                        "priority": "recommended",
                        "reasoning": "Would elevate the look",
                        "style_suggestion": self._suggest_style_for_item(nice_item)
                    })
        
        # Generate ideal outfit description
        ideal_outfit = self._create_ideal_outfit_description(context)
        
        return {
            "has_gaps": True,
            "avg_relevancy": avg_relevancy,
            "gaps": gaps[:5],  # Limit to top 5 suggestions
            "ideal_outfit": ideal_outfit,
            "message": self._format_gap_message(gaps[:5], ideal_outfit, context)
        }
    
    def _suggest_style_for_item(self, item: str) -> str:
        """Suggest style for missing item based on user's profile."""
        style = self.user_context.get("inferred_style", "classic")
        colors = self.user_context.get("color_preferences", [])
        
        # Get first color or default
        color_desc = "neutral"
        if colors:
            color_map = {
                "#000000": "black", "#FFFFFF": "white", "#808080": "gray",
                "#1E3A8A": "navy", "#1560BD": "blue"
            }
            color_desc = color_map.get(colors[0], "neutral")
        
        # Item-specific suggestions
        suggestions = {
            "blazer": f"{style} blazer in {color_desc} or navy",
            "dress shirt": f"crisp {color_desc if color_desc == 'white' else 'white'} dress shirt",
            "dress pants": f"{style} {color_desc if color_desc in ['black', 'navy', 'gray'] else 'black'} trousers",
            "dress shoes": f"{color_desc if color_desc == 'black' else 'black'} {style} dress shoes",
            "formal dress": f"{style} {color_desc} formal dress or gown",
            "heels": f"{color_desc if color_desc in ['black', 'nude'] else 'black'} heels (3-4 inch)",
            "athletic top": f"{style} moisture-wicking athletic top",
            "athletic bottoms": f"{style} performance leggings or shorts",
            "dressy top": f"{style} {color_desc} dressy blouse or top",
            "casual top": f"{style} casual {color_desc} top or tee"
        }
        
        return suggestions.get(item.lower(), f"{style} {item}")
    
    def _create_ideal_outfit_description(self, context: Dict) -> str:
        """Generate ideal outfit description using Gemini."""
        try:
            prompt = f"""Based on this user's style profile:

Style: {self.user_context.get('inferred_style', 'classic')}
Personality: {', '.join(self.user_context.get('personality_traits', ['stylish', 'modern']))}
Lifestyle: {self.user_context.get('lifestyle', 'contemporary professional')}
Favorite colors: {', '.join([c for c in self.user_context.get('color_preferences', [])]) or 'versatile neutrals'}

Current situation:
Event: {context.get('event_type', 'special occasion')}
Weather: {context.get('weather', 'moderate temperature')}
Desired feeling: {context.get('desired_feeling', 'confident and appropriate')}

Describe the IDEAL outfit for this person in 3-4 sentences. Be specific about:
- Exact clothing items (not just "top" but "silk blouse" or "structured blazer")
- Colors that suit their preferences
- Why it works for their personal style
- How it fits the occasion

Keep it concise, specific, and personalized."""
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part(text=prompt)],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=200)
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error creating ideal outfit: {e}")
            # Fallback description
            return f"A polished {self.user_context.get('inferred_style', 'classic')} outfit perfect for {context.get('event_type', 'the occasion')}, incorporating your favorite colors and style preferences."
    
    def _format_gap_message(self, gaps: List[Dict], ideal_outfit: str, context: Dict) -> str:
        """Format the wardrobe gap message for voice output."""
        essential_gaps = [g for g in gaps if g.get('priority') == 'essential']
        recommended_gaps = [g for g in gaps if g.get('priority') == 'recommended']
        
        if not essential_gaps and recommended_gaps:
            # Only nice-to-have items missing
            return f"""Your closet has some great versatile pieces! Adding a few items would really complete your look for {context.get('event_type', 'this occasion')}.

The ideal outfit would be: {ideal_outfit}

Here's what would take your style to the next level:
{chr(10).join(f"• {g['category'].title()} - {g['style_suggestion']}" for g in recommended_gaps[:3])}

These pieces would perfectly complement your {self.user_context.get('inferred_style', 'personal')} style!"""
        
        # Essential items missing
        return f"""I've looked through your closet, and while you have some great pieces, your wardrobe has a few gaps for {context.get('event_type', 'this occasion')}.

The ideal look would be: {ideal_outfit}

Here's what I'd suggest shopping for:
{chr(10).join(f"• {g['category'].title()} - {g['reasoning']}" for g in essential_gaps)}

Based on your {self.user_context.get('inferred_style', 'personal')} style and lifestyle, these would be perfect additions!"""


    # ============================================================================
    # ADDITIONAL PRIORITY SCENARIOS
    # ============================================================================
    
    def check_for_duplicates(self, items: List[Dict]) -> List[Dict]:
        """
        SCENARIO 7: Remove duplicate/very similar items.
        
        Prevents recommending multiple identical items (e.g., 5 black t-shirts).
        Keeps the best version of each similar item.
        
        Args:
            items: List of closet items
            
        Returns:
            Deduplicated list with best items kept
        """
        if len(items) <= 1:
            return items
        
        unique_items = []
        seen_signatures = set()
        
        for item in items:
            # Create signature: category + color + style
            category = str(item.get('category', 'unknown')).lower()
            color = str(item.get('color_hex', 'unknown')).lower()
            style = str(item.get('style', 'unknown')).lower()
            
            signature = f"{category}_{color}_{style}"
            
            if signature not in seen_signatures:
                unique_items.append(item)
                seen_signatures.add(signature)
            # else: skip duplicate
        
        return unique_items
    
    def handle_multi_event_day(self, events: List[Dict]) -> Dict:
        """
        SCENARIO 10: Multi-event day recommendations.
        
        Handles "office meeting then dinner date" scenarios.
        
        Args:
            events: List of [{type, formality, time}]
            
        Returns:
            Recommendation with transition strategy
        """
        if len(events) <= 1:
            # Single event - use normal recommendation
            return None
        
        # Analyze formality levels
        formality_scores = {
            "formal": 5,
            "business": 4,
            "smart casual": 3,
            "casual": 2,
            "athletic": 1
        }
        
        max_formality = max(formality_scores.get(e.get('formality', 'casual').lower(), 2) for e in events)
        min_formality = min(formality_scores.get(e.get('formality', 'casual').lower(), 2) for e in events)
        
        formality_gap = max_formality - min_formality
        
        if formality_gap >= 2:
            # Large formality gap - suggest outfit change
            return {
                "strategy": "outfit_change",
                "message": f"Since you're going from {events[0]['type']} to {events[1]['type']}, I'd recommend changing outfits. The formality levels are quite different.",
                "outfit_1": {"for": events[0]['type']},
                "outfit_2": {"for": events[1]['type']},
                "transition_tip": "Keep a change of clothes in your bag or car"
            }
        else:
            # Can transition with same base outfit
            return {
                "strategy": "transition_outfit",
                "message": f"Perfect! I can create one outfit that transitions from {events[0]['type']} to {events[1]['type']}.",
                "base_outfit": "smart casual foundation",
                "transition_tip": "Add accessories or remove blazer to adjust formality",
                "target_formality": "smart casual"  # Middle ground
            }
    
    def resolve_vlm_closet_conflict(self) -> None:
        """
        SCENARIO 16: Resolve VLM/closet conflicts.
        
        If VLM shows gym scenes but closet has only formal wear,
        trust the closet (current state) over VLM (past context).
        """
        vlm_history = self.user_context.get('vlm_history', [])
        
        if not vlm_history or not self.user_closet:
            return
        
        # Analyze VLM scene types
        vlm_vibes = {}
        for vlm in vlm_history:
            scene_style = vlm.get('overall_style', '').lower()
            if 'athletic' in scene_style or 'gym' in scene_style or 'sporty' in scene_style:
                vlm_vibes['athletic'] = vlm_vibes.get('athletic', 0) + 1
            elif 'formal' in scene_style or 'business' in scene_style:
                vlm_vibes['formal'] = vlm_vibes.get('formal', 0) + 1
            elif 'casual' in scene_style:
                vlm_vibes['casual'] = vlm_vibes.get('casual', 0) + 1
        
        # Analyze closet vibes
        closet_vibes = {}
        for item in self.user_closet:
            vibe = str(item.get('occasion_vibe', '')).lower()
            if 'athletic' in vibe or 'sporty' in vibe or 'gym' in vibe:
                closet_vibes['athletic'] = closet_vibes.get('athletic', 0) + 1
            elif 'formal' in vibe or 'business' in vibe:
                closet_vibes['formal'] = closet_vibes.get('formal', 0) + 1
            elif 'casual' in vibe:
                closet_vibes['casual'] = closet_vibes.get('casual', 0) + 1
        
        # Detect major conflicts
        dominant_vlm = max(vlm_vibes, key=vlm_vibes.get) if vlm_vibes else None
        dominant_closet = max(closet_vibes, key=closet_vibes.get) if closet_vibes else None
        
        if dominant_vlm and dominant_closet and dominant_vlm != dominant_closet:
            # Conflict detected - update context to trust closet
            print(f"⚠️  VLM/Closet conflict: VLM suggests {dominant_vlm}, closet is {dominant_closet}")
            print(f"    → Trusting current closet state")
            
            # Weight recent uploads higher (trust current state)
            # Filter out conflicting VLM history
            filtered_vlm = [
                v for v in vlm_history 
                if dominant_closet in str(v.get('overall_style', '')).lower()
            ]
            
            if filtered_vlm:
                self.user_context['vlm_history'] = filtered_vlm[-5:]  # Keep 5 most relevant
    
    def avoid_repetition(self, new_outfit: Dict) -> bool:
        """
        SCENARIO 18: Avoid recommending same outfit repeatedly.
        
        Checks if outfit was recently recommended.
        
        Args:
            new_outfit: Proposed outfit dict
            
        Returns:
            True if outfit is too similar to recent recommendation
        """
        if not self.recommendation_history:
            return False
        
        # Check last 3 recommendations
        recent = self.recommendation_history[-3:]
        
        for past_rec in recent:
            past_outfit = past_rec.get('outfit', {})
            
            # Count matching items
            matches = 0
            total_items = 0
            
            for category in ['top', 'bottom', 'shoes', 'outerwear']:
                if category in new_outfit.get('outfit', {}):
                    total_items += 1
                    new_item_id = new_outfit['outfit'][category].get('id')
                    past_item_id = past_outfit.get(category, {}).get('id')
                    
                    if new_item_id == past_item_id:
                        matches += 1
            
            # If 75%+ items match, it's too repetitive
            if total_items > 0 and (matches / total_items) >= 0.75:
                return True
        
        return False
    
    def save_recommendation(self, recommendation: Dict) -> None:
        """
        SCENARIO 18: Save recommendation to history.
        
        Args:
            recommendation: Outfit recommendation dict
        """
        from datetime import datetime
        
        self.recommendation_history.append({
            'outfit': recommendation.get('outfit', {}),
            'timestamp': datetime.now().isoformat(),
            'event_type': self.user_context.get('event_type'),
            'weather': self.user_context.get('weather')
        })
        
        # Keep only last 10 recommendations
        if len(self.recommendation_history) > 10:
            self.recommendation_history = self.recommendation_history[-10:]
    
    def validate_color_coordination(self, outfit: Dict) -> Dict:
        """
        SCENARIO 20: Validate color coordination.
        
        Checks for fashion faux pas:
        - Navy + black (clashing dark colors)
        - Brown + black (generally avoid)
        - Too many competing colors (>3 main colors)
        
        Args:
            outfit: Outfit dict with top, bottom, shoes
            
        Returns:
            Dict with validation results and suggestions
        """
        issues = []
        suggestions = []
        
        # Extract colors from outfit
        colors = []
        color_items = {}
        
        for category in ['top', 'bottom', 'shoes', 'outerwear']:
            if category in outfit:
                item_id = outfit[category].get('id')
                item = next((i for i in self.user_closet if i['id'] == item_id), None)
                
                if item:
                    color_hex = item.get('color_hex', '')
                    if color_hex:
                        colors.append(color_hex)
                        color_items[color_hex] = color_items.get(color_hex, []) + [category]
        
        # Color clash detection
        has_black = '#000000' in colors or any(c.lower().startswith('#0000') for c in colors)
        has_navy = '#1E3A8A' in colors or '#000080' in colors or any('3a8a' in c.lower() or '0080' in c.lower() for c in colors)
        has_brown = any('8b4513' in c.lower() or 'a52a2a' in c.lower() or '654321' in c.lower() for c in colors)
        
        # Rule 1: Navy + Black (unless intentional monochrome)
        if has_black and has_navy:
            issues.append("Navy and black together can clash")
            suggestions.append("Try replacing one with gray, charcoal, or another neutral")
        
        # Rule 2: Brown + Black
        if has_black and has_brown:
            issues.append("Brown and black together is generally avoided")
            suggestions.append("Replace black with navy or gray, or brown with tan/beige")
        
        # Rule 3: Too many colors (>3 distinct non-neutral colors)
        neutrals = ['#FFFFFF', '#000000', '#808080', '#C0C0C0', '#F5F5DC', '#FFFAF0']
        non_neutral_colors = [c for c in colors if c not in neutrals and not any(n in c for n in neutrals)]
        
        if len(non_neutral_colors) > 3:
            issues.append(f"Too many competing colors ({len(non_neutral_colors)} different colors)")
            suggestions.append("Stick to 2-3 main colors for a cohesive look")
        
        # Rule 4: Formality mismatch (casual color with formal item)
        # Check if bright casual colors paired with formal items
        bright_casual_colors = ['#FF0000', '#FFFF00', '#00FF00', '#FFA500']  # Red, yellow, green, orange
        
        for color in colors:
            if any(bright in color.upper() for bright in bright_casual_colors):
                categories_with_color = color_items.get(color, [])
                if any(cat in ['blazer', 'dress pants', 'dress shoes'] for cat in categories_with_color):
                    issues.append("Bright casual color on formal item")
                    suggestions.append("Use muted tones for formal pieces")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "colors_used": colors
        }
    
    def handle_missing_images(self, items: List[Dict]) -> List[Dict]:
        """
        SCENARIO 14: Handle missing or broken image URLs.
        
        Filters out items with missing images or provides placeholders.
        
        Args:
            items: List of closet items
            
        Returns:
            List of items with valid images
        """
        valid_items = []
        
        for item in items:
            image_path = item.get('image_path') or item.get('image_url')
            
            # Check if image exists and is valid
            if not image_path:
                print(f"⚠️  Item {item.get('id')} has no image, skipping from collage")
                continue
            
            # Check if it's a valid format
            if image_path.startswith('data:image') or image_path.startswith('http'):
                valid_items.append(item)
            else:
                # Try to construct full URL if it's a path
                if image_path.startswith('/'):
                    full_url = f"{self.closet_api_url}{image_path}"
                    item_copy = item.copy()
                    item_copy['image_path'] = full_url
                    valid_items.append(item_copy)
                else:
                    print(f"⚠️  Item {item.get('id')} has invalid image format: {image_path[:50]}")
                    continue
        
        return valid_items
    
    def handle_no_similar_items(self, inspo_analysis: Dict) -> Dict:
        """
        SCENARIO 12: Handle case when no similar items found.
        
        Provides helpful feedback and shopping suggestions.
        
        Args:
            inspo_analysis: Inspiration analysis dict
            
        Returns:
            Response dict with explanation and suggestions
        """
        return {
            "found_matches": False,
            "message": f"I love the {inspo_analysis.get('style', 'style')} vibe of this inspiration! Unfortunately, I couldn't find very similar items in your current closet.",
            "inspiration_style": inspo_analysis.get('style'),
            "inspiration_colors": inspo_analysis.get('colors', []),
            "closest_items": self._find_closest_partial_matches(inspo_analysis),
            "shopping_suggestions": self._generate_inspo_shopping_list(inspo_analysis),
            "creative_adaptation": self._suggest_creative_adaptation(inspo_analysis)
        }
    
    def _find_closest_partial_matches(self, inspo: Dict) -> List[Dict]:
        """Find items with ANY similarity (lower threshold)."""
        partial_matches = []
        
        for item in self.user_closet:
            score = self._calculate_inspiration_similarity(inspo, item)
            if score >= 40:  # Lower threshold - any partial match
                partial_matches.append({
                    "item": item,
                    "score": round(score, 1),
                    "reasoning": f"Partially matches {inspo.get('style', 'style')}"
                })
        
        return sorted(partial_matches, key=lambda x: x['score'], reverse=True)[:3]
    
    def _generate_inspo_shopping_list(self, inspo: Dict) -> List[str]:
        """Generate shopping list based on inspiration."""
        items_needed = inspo.get('items', [])
        style = inspo.get('style', 'stylish')
        colors = inspo.get('colors', [])
        
        shopping_list = []
        for item in items_needed[:3]:  # Top 3 items
            color_desc = f" in {colors[0]}" if colors else ""
            shopping_list.append(f"{style} {item}{color_desc}")
        
        return shopping_list
    
    def _suggest_creative_adaptation(self, inspo: Dict) -> str:
        """Suggest how to adapt inspiration with current closet."""
        style = inspo.get('style', 'this style')
        
        return f"To achieve {style} with your current wardrobe, focus on the vibe and silhouette rather than exact pieces. Layer items creatively and use accessories to capture the aesthetic!"


# ==========================================
# USAGE EXAMPLE
# ==========================================

if __name__ == "__main__":
    # Initialize stylist
    stylist = VoicePersonalStylist(
        closet_api_url="http://localhost:8000",
        auth_token="your_jwt_token_here",  # Get from /token endpoint
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="EXAVITQu4vr4xnSDxMaL"  # Bella voice (warm, friendly)
    )
    
    # Start voice styling session
    stylist.start_styling_session()
