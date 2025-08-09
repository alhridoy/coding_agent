#!/usr/bin/env python3
"""
Interactive demo script for the Autonomous Coding Agent
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import httpx
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import get_settings


class DemoRunner:
    """Interactive demo runner for the autonomous coding agent"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.example_repos = [
            {
                "name": "Simple Flask App",
                "url": "https://github.com/pallets/flask",
                "prompts": [
                    "Add a health check endpoint that returns API status",
                    "Add request logging middleware",
                    "Add rate limiting to API endpoints"
                ]
            },
            {
                "name": "Express.js App", 
                "url": "https://github.com/expressjs/express",
                "prompts": [
                    "Add input validation middleware",
                    "Add error handling middleware",
                    "Add security headers middleware"
                ]
            }
        ]
    
    def print_banner(self):
        """Print the demo banner"""
        print("🤖 " + "="*60)
        print("🤖 AUTONOMOUS CODING AGENT DEMO")
        print("🤖 " + "="*60)
        print()
        print("This demo will help you test the autonomous coding agent")
        print("by making real API requests and showing streaming results.")
        print()
    
    def check_environment(self) -> bool:
        """Check if environment is properly configured"""
        print("🔍 Checking environment configuration...")
        
        try:
            settings = get_settings()
            
            # Check required API keys
            missing_keys = []
            if not settings.anthropic_api_key:
                missing_keys.append("ANTHROPIC_API_KEY")
            if not settings.github_token:
                missing_keys.append("GITHUB_TOKEN")
            
            if missing_keys:
                print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
                print("\nPlease set these in your .env file:")
                for key in missing_keys:
                    print(f"  {key}=your_key_here")
                return False
            
            print("✅ Environment configuration looks good!")
            return True
            
        except Exception as e:
            print(f"❌ Error checking environment: {e}")
            return False
    
    async def check_server_health(self) -> bool:
        """Check if the server is running"""
        print("🏥 Checking server health...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    print("✅ Server is healthy!")
                    print(f"   Status: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Server health check failed: {response.status_code}")
                    return False
                    
        except httpx.ConnectError:
            print("❌ Cannot connect to server. Is it running?")
            print("   Start the server with: python -m src.api.main")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def select_test_scenario(self) -> Dict[str, Any]:
        """Let user select a test scenario"""
        print("\n📋 Available test scenarios:")
        print("1. Use example repository (recommended)")
        print("2. Use your own repository")
        print("3. Quick test with minimal prompt")
        
        while True:
            try:
                choice = input("\nSelect option (1-3): ").strip()
                
                if choice == "1":
                    return self.select_example_repo()
                elif choice == "2":
                    return self.get_custom_repo()
                elif choice == "3":
                    return self.get_quick_test()
                else:
                    print("Please select 1, 2, or 3")
                    
            except KeyboardInterrupt:
                print("\nDemo cancelled.")
                sys.exit(0)
    
    def select_example_repo(self) -> Dict[str, Any]:
        """Select an example repository"""
        print("\n📦 Example repositories:")
        
        for i, repo in enumerate(self.example_repos, 1):
            print(f"{i}. {repo['name']}")
            print(f"   URL: {repo['url']}")
            print(f"   Sample prompts: {len(repo['prompts'])}")
            print()
        
        while True:
            try:
                choice = int(input(f"Select repository (1-{len(self.example_repos)}): ")) - 1
                
                if 0 <= choice < len(self.example_repos):
                    repo = self.example_repos[choice]
                    
                    # Select prompt
                    print(f"\n💡 Available prompts for {repo['name']}:")
                    for i, prompt in enumerate(repo['prompts'], 1):
                        print(f"{i}. {prompt}")
                    
                    while True:
                        try:
                            prompt_choice = int(input(f"Select prompt (1-{len(repo['prompts'])}): ")) - 1
                            
                            if 0 <= prompt_choice < len(repo['prompts']):
                                return {
                                    "repo_url": repo['url'],
                                    "prompt": repo['prompts'][prompt_choice],
                                    "branch_name": f"demo/{repo['prompts'][prompt_choice].lower().replace(' ', '-')[:20]}"
                                }
                            else:
                                print(f"Please select 1-{len(repo['prompts'])}")
                                
                        except ValueError:
                            print("Please enter a number")
                            
                else:
                    print(f"Please select 1-{len(self.example_repos)}")
                    
            except ValueError:
                print("Please enter a number")
    
    def get_custom_repo(self) -> Dict[str, Any]:
        """Get custom repository details"""
        print("\n🔧 Custom repository setup:")
        
        repo_url = input("Enter GitHub repository URL: ").strip()
        if not repo_url.startswith("https://github.com/"):
            print("Warning: URL should start with https://github.com/")
        
        prompt = input("Enter coding prompt: ").strip()
        
        branch_name = input("Enter branch name (optional): ").strip()
        if not branch_name:
            branch_name = f"demo/auto-{prompt.lower().replace(' ', '-')[:20]}"
        
        return {
            "repo_url": repo_url,
            "prompt": prompt,
            "branch_name": branch_name
        }
    
    def get_quick_test(self) -> Dict[str, Any]:
        """Get quick test configuration"""
        return {
            "repo_url": "https://github.com/pallets/flask",
            "prompt": "Add a simple health check endpoint",
            "branch_name": "demo/quick-test"
        }
    
    async def run_coding_session(self, config: Dict[str, Any]):
        """Run a coding session with the given configuration"""
        print("\n🚀 Starting coding session...")
        print(f"   Repository: {config['repo_url']}")
        print(f"   Prompt: {config['prompt']}")
        print(f"   Branch: {config['branch_name']}")
        print()
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/code",
                    json=config,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status_code != 200:
                        print(f"❌ Request failed: {response.status_code}")
                        error_text = await response.aread()
                        print(f"   Error: {error_text.decode()}")
                        return
                    
                    print("📡 Streaming events:")
                    print("-" * 50)
                    
                    event_count = 0
                    async for chunk in response.aiter_text():
                        for line in chunk.split('\n'):
                            if line.strip().startswith('data: '):
                                event_count += 1
                                try:
                                    event_data = json.loads(line[6:])  # Remove 'data: '
                                    self.display_event(event_data, event_count)
                                except json.JSONDecodeError:
                                    print(f"[{event_count}] Raw: {line}")
                    
                    print("-" * 50)
                    print(f"✅ Session completed! Received {event_count} events")
                    
        except httpx.TimeoutException:
            print("❌ Request timed out. The operation might still be running.")
        except Exception as e:
            print(f"❌ Error during coding session: {e}")
    
    def display_event(self, event_data: Dict[str, Any], event_num: int):
        """Display a streaming event"""
        event_type = event_data.get('type', 'Unknown')
        
        # Color coding for different event types
        color_map = {
            'Status': '🔄',
            'AI Message': '🧠',
            'Tool: Read': '📖',
            'Tool: Edit': '✏️',
            'Tool: Bash': '⚡',
            'PR Created': '🎉',
            'Error': '❌'
        }
        
        icon = color_map.get(event_type, '📋')
        
        if event_type == 'Status':
            print(f"{icon} [{event_num}] {event_data.get('message', '')}")
            
        elif event_type == 'AI Message':
            message = event_data.get('message', '')
            print(f"{icon} [{event_num}] AI: {message}")
            
        elif event_type.startswith('Tool:'):
            if event_type == 'Tool: Read':
                filepath = event_data.get('filepath', '')
                print(f"{icon} [{event_num}] Reading: {filepath}")
            elif event_type == 'Tool: Edit':
                filepath = event_data.get('filepath', '')
                print(f"{icon} [{event_num}] Editing: {filepath}")
            elif event_type == 'Tool: Bash':
                command = event_data.get('command', '')
                print(f"{icon} [{event_num}] Running: {command}")
            
        elif 'pr_url' in event_data:
            pr_url = event_data.get('pr_url', '')
            print(f"{icon} [{event_num}] PR Created: {pr_url}")
            
        elif event_type == 'Error':
            error_msg = event_data.get('error_message', '')
            print(f"{icon} [{event_num}] Error: {error_msg}")
            
        else:
            # Display raw event for unknown types
            print(f"📋 [{event_num}] {event_type}: {event_data}")
    
    async def run_demo(self):
        """Run the complete demo"""
        self.print_banner()
        
        # Check environment
        if not self.check_environment():
            return
        
        # Check server health
        if not await self.check_server_health():
            return
        
        # Get test configuration
        config = self.select_test_scenario()
        
        # Confirm before running
        print("\n📝 Configuration summary:")
        print(f"   Repository: {config['repo_url']}")
        print(f"   Prompt: {config['prompt']}")
        print(f"   Branch: {config['branch_name']}")
        
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Demo cancelled.")
            return
        
        # Run the coding session
        await self.run_coding_session(config)
        
        print("\n🎉 Demo completed!")
        print("   Check the PR URL (if provided) to see the changes.")
        print("   Visit http://localhost:8000/health to verify the server is still running.")


async def main():
    """Main demo entry point"""
    try:
        demo = DemoRunner()
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())