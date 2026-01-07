import logging
from typing import List, Dict, Optional
import requests
import time
from .offline_data import OFFLINE_POSTS

# Configure module logger
logger = logging.getLogger(__name__)


class Post:
    """Container for post data."""
    
    def __init__(self, post_id: int, title: str, body: str, user_id: int):
        self.id = post_id
        self.title = title
        self.body = body
        self.user_id = user_id
    
    def format_content(self) -> str:
        """
        Format post content for text file.
        
        Returns:
            str: Formatted post content
        """
        content = f"Title: {self.title}\n\n"
        content += f"{self.body}\n"
        return content
    
    def __repr__(self):
        return f"Post(id={self.id}, title='{self.title[:30]}...')"


class DataProvider:
    """
    API client for fetching posts from JSONPlaceholder.
    
    Provides data validation, error handling, and timeout management.
    """
    
    API_URL = "https://jsonplaceholder.typicode.com/posts"
    
    def __init__(self, timeout: int = 10):
        """
        Initialize data provider.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        logger.info("DataProvider initialized")
    
    def fetch_posts(self, limit: int = 10) -> List[Post]:
        """
        Fetch posts from JSONPlaceholder API with retry and offline fallback.
        
        Tries 3 times to fetch from API. If all attempts fail, switches to offline mode.
        
        Args:
            limit: Maximum number of posts to fetch
            
        Returns:
            List[Post]: List of validated Post objects
        """
        max_retries = 3
        retry_delay = 2  # seconds
        
        # Try API connection up to 3 times
        for attempt in range(1, max_retries + 1):
            logger.info(f"Fetching {limit} posts from API (attempt {attempt}/{max_retries})...")
            
            try:
                # Make API request
                response = requests.get(
                    self.API_URL,
                    timeout=self.timeout,
                    params={'_limit': limit}
                )
                
                # Check response status
                response.raise_for_status()
                
                # Parse JSON
                posts_data = response.json()
                
                if not isinstance(posts_data, list):
                    raise ValueError("API response is not a list")
                
                logger.info(f"Received {len(posts_data)} posts from API")
                
                # Validate and convert to Post objects
                posts = []
                for post_data in posts_data:
                    post = self._validate_and_create_post(post_data)
                    if post:
                        posts.append(post)
                
                logger.info(f"✓ Successfully processed {len(posts)} valid posts from API")
                return posts
                
            except requests.exceptions.Timeout:
                logger.warning(f"Attempt {attempt}/{max_retries}: API request timed out after {self.timeout}s")
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Attempt {attempt}/{max_retries}: Connection error: {e}")
                
            except requests.exceptions.HTTPError as e:
                logger.warning(f"Attempt {attempt}/{max_retries}: HTTP error: {e}")
                
            except ValueError as e:
                logger.warning(f"Attempt {attempt}/{max_retries}: Invalid API response: {e}")
                
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries}: Unexpected error: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
        
        # All API attempts failed - switch to offline mode
        logger.warning("="*60)
        logger.warning("⚠️  API CONNECTION FAILED AFTER 3 ATTEMPTS")
        logger.warning("🔄 SWITCHING TO OFFLINE MODE")
        logger.warning("="*60)
        
        return self._load_offline_posts(limit)
    
    def _load_offline_posts(self, limit: int) -> List[Post]:
        """
        Load posts from offline data.
        
        Args:
            limit: Maximum number of posts to load
            
        Returns:
            List[Post]: List of Post objects from offline data
        """
        logger.info(f"Loading {limit} posts from offline data...")
        
        try:
            posts = []
            offline_data = OFFLINE_POSTS[:limit]
            
            for post_data in offline_data:
                post = self._validate_and_create_post(post_data)
                if post:
                    posts.append(post)
            
            logger.info(f"✓ Successfully loaded {len(posts)} posts from offline data")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to load offline data: {e}")
            return []
    
    def _validate_and_create_post(self, post_data: Dict) -> Optional[Post]:
        """
        Validate post data and create Post object.
        
        Args:
            post_data: Raw post data dictionary from API
            
        Returns:
            Post: Validated Post object, or None if invalid
        """
        try:
            # Check required fields
            required_fields = ['id', 'userId', 'title', 'body']
            for field in required_fields:
                if field not in post_data:
                    logger.warning(f"Post missing field '{field}': {post_data}")
                    return None
            
            # Extract and validate data
            post_id = post_data['id']
            user_id = post_data['userId']
            title = post_data['title'].strip()
            body = post_data['body'].strip()
            
            # Validate types
            if not isinstance(post_id, int) or not isinstance(user_id, int):
                logger.warning(f"Invalid post ID types: {post_data}")
                return None
            
            if not title or not body:
                logger.warning(f"Post has empty title or body: {post_id}")
                return None
            
            # Create Post object
            post = Post(
                post_id=post_id,
                title=title,
                body=body,
                user_id=user_id
            )
            
            logger.debug(f"Validated post: {post}")
            return post
            
        except Exception as e:
            logger.error(f"Error validating post data: {e}")
            return None
    
    def fetch_single_post(self, post_id: int) -> Optional[Post]:
        """
        Fetch a single post by ID.
        
        Args:
            post_id: Post ID to fetch
            
        Returns:
            Post: Post object if found, None otherwise
        """
        logger.info(f"Fetching post {post_id}...")
        
        try:
            url = f"{self.API_URL}/{post_id}"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            post_data = response.json()
            post = self._validate_and_create_post(post_data)
            
            if post:
                logger.info(f"✓ Successfully fetched post {post_id}")
            else:
                logger.warning(f"Invalid post data for ID {post_id}")
            
            return post
            
        except Exception as e:
            logger.error(f"Error fetching post {post_id}: {e}")
            return None


def test_api_connection() -> bool:
    """
    Test API connectivity and response.
    
    Returns:
        bool: True if API is accessible, False otherwise
    """
    logger.info("Testing API connection...")
    
    try:
        provider = DataProvider(timeout=5)
        posts = provider.fetch_posts(limit=1)
        
        if posts and len(posts) > 0:
            logger.info("✓ API connection test successful")
            return True
        else:
            logger.warning("API responded but returned no data")
            return False
            
    except Exception as e:
        logger.error(f"API connection test failed: {e}")
        return False
