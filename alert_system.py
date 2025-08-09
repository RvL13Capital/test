# project/alert_system.py
"""
Multi-channel alert and notification system for trading signals
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from .signal_intelligence_hub import TradingSignal, SignalStrength
from .storage import get_gcs_storage
from .config import Config

logger = logging.getLogger(__name__)

@dataclass
class AlertPreferences:
    """User alert preferences"""
    user_id: str
    email_enabled: bool = True
    slack_enabled: bool = False
    webhook_enabled: bool = False
    desktop_enabled: bool = False
    
    email_address: Optional[str] = None
    slack_webhook: Optional[str] = None
    webhook_url: Optional[str] = None
    
    min_signal_strength: SignalStrength = SignalStrength.MODERATE
    specific_tickers: Optional[List[str]] = None
    immediate_notify: bool = True
    daily_summary: bool = True
    
    quiet_hours_start: Optional[int] = None  # Hour in 24h format
    quiet_hours_end: Optional[int] = None

class AlertChannel:
    """Base class for alert channels"""
    
    async def send(self, alert: Dict, user_id: str) -> bool:
        """Send alert through channel"""
        raise NotImplementedError

class EmailAlertChannel(AlertChannel):
    """Email alert channel"""
    
    def __init__(self):
        self.smtp_server = Config.SMTP_SERVER
        self.smtp_port = Config.SMTP_PORT
        self.smtp_user = Config.SMTP_USER
        self.smtp_pass = Config.SMTP_PASSWORD
        self.from_email = Config.FROM_EMAIL
    
    async def send(self, alert: Dict, user_id: str) -> bool:
        """Send email alert"""
        try:
            # Get user email
            preferences = AlertSystem.get_user_preferences_static(user_id)
            to_email = preferences.get('email_address')
            
            if not to_email:
                logger.warning(f"No email address for user {user_id}")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = alert['title']
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            # Create HTML content
            html = self._create_html_email(alert)
            
            # Create plain text content
            text = alert['message']
            
            part1 = MIMEText(text, 'plain')
            part2 = MIMEText(html, 'html')
            
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_html_email(self, alert: Dict) -> str:
        """Create HTML email content"""
        
        data = alert.get('data', {})
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .signal-box {{ 
                    border: 2px solid #3498db; 
                    border-radius: 10px; 
                    padding: 15px; 
                    margin: 20px 0;
                }}
                .metrics {{ 
                    display: grid; 
                    grid-template-columns: repeat(2, 1fr); 
                    gap: 10px;
                    margin: 15px 0;
                }}
                .metric {{ 
                    padding: 10px; 
                    background: #ecf0f1; 
                    border-radius: 5px;
                }}
                .strong {{ color: #27ae60; font-weight: bold; }}
                .warning {{ color: #e74c3c; }}
                .button {{
                    background-color: #3498db;
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{alert['title']}</h1>
            </div>
            <div class="content">
                <div class="signal-box">
                    <h2>Signal Details</h2>
                    <pre>{alert['message']}</pre>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <strong>Ticker:</strong> {data.get('ticker', 'N/A')}
                    </div>
                    <div class="metric">
                        <strong>Confidence:</strong> {data.get('confidence', 0):.1%}
                    </div>
                    <div class="metric">
                        <strong>Entry:</strong> ${data.get('entry_price', 0):.2f}
                    </div>
                    <div class="metric">
                        <strong>Stop Loss:</strong> ${data.get('stop_loss', 0):.2f}
                    </div>
                </div>
                
                <a href="{Config.DASHBOARD_URL}/signals/{data.get('signal_id', '')}" class="button">
                    View in Dashboard
                </a>
                
                <p style="color: #7f8c8d; font-size: 12px; margin-top: 30px;">
                    This is an automated alert from your trading signal system. 
                    Please do your own research before making any trading decisions.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html

class SlackAlertChannel(AlertChannel):
    """Slack alert channel"""
    
    async def send(self, alert: Dict, user_id: str) -> bool:
        """Send Slack alert"""
        try:
            preferences = AlertSystem.get_user_preferences_static(user_id)
            webhook_url = preferences.get('slack_webhook')
            
            if not webhook_url:
                logger.warning(f"No Slack webhook for user {user_id}")
                return False
            
            # Format for Slack
            slack_message = self._format_slack_message(alert)
            
            response = requests.post(webhook_url, json=slack_message)
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent to {user_id}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _format_slack_message(self, alert: Dict) -> Dict:
        """Format message for Slack"""
        
        data = alert.get('data', {})
        
        return {
            "text": alert['title'],
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": alert['title']
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert['message']
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Ticker:* {data.get('ticker', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Confidence:* {data.get('confidence', 0):.1%}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Entry:* ${data.get('entry_price', 0):.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Risk/Reward:* {data.get('risk_reward_ratio', 0):.1f}:1"
                        }
                    ]
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "View Details"
                            },
                            "url": f"{Config.DASHBOARD_URL}/signals/{data.get('signal_id', '')}"
                        }
                    ]
                }
            ]
        }

class WebhookAlertChannel(AlertChannel):
    """Generic webhook alert channel"""
    
    async def send(self, alert: Dict, user_id: str) -> bool:
        """Send webhook alert"""
        try:
            preferences = AlertSystem.get_user_preferences_static(user_id)
            webhook_url = preferences.get('webhook_url')
            
            if not webhook_url:
                logger.warning(f"No webhook URL for user {user_id}")
                return False
            
            # Send webhook
            response = requests.post(
                webhook_url,
                json={
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'alert': alert
                },
                timeout=10
            )
            
            if response.status_code in [200, 201, 202, 204]:
                logger.info(f"Webhook alert sent to {user_id}")
                return True
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

class DesktopNotificationChannel(AlertChannel):
    """Desktop notification channel (requires client-side implementation)"""
    
    async def send(self, alert: Dict, user_id: str) -> bool:
        """Send desktop notification"""
        try:
            # This would typically send to a WebSocket connection
            # or use a service like Pusher/Firebase
            
            logger.info(f"Desktop notification queued for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")
            return False

class AlertSystem:
    """Main alert system coordinator"""
    
    def __init__(self):
        self.channels = {
            'email': EmailAlertChannel(),
            'slack': SlackAlertChannel(),
            'webhook': WebhookAlertChannel(),
            'desktop': DesktopNotificationChannel()
        }
        
        self.user_preferences = {}
        self.alert_history = []
        self.gcs = get_gcs_storage()
        
        # Load user preferences from storage
        self._load_user_preferences()
        
        logger.info("Alert system initialized")
    
    def _load_user_preferences(self):
        """Load user preferences from storage"""
        try:
            if self.gcs:
                # Load preferences from GCS
                # In production, implement actual loading
                pass
        except Exception as e:
            logger.error(f"Failed to load user preferences: {e}")
    
    @staticmethod
    def get_user_preferences_static(user_id: str) -> Dict:
        """Static method to get user preferences"""
        # In production, query from database
        return {
            'email_address': f"{user_id}@example.com",
            'slack_webhook': None,
            'webhook_url': None
        }
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user alert preferences"""
        
        if user_id not in self.user_preferences:
            # Default preferences
            self.user_preferences[user_id] = {
                'channels': ['email'],
                'min_strength': SignalStrength.MODERATE.value,
                'email_address': None,
                'slack_webhook': None,
                'webhook_url': None,
                'immediate_notify': True,
                'daily_summary': True,
                'quiet_hours_start': None,
                'quiet_hours_end': None,
                'specific_tickers': None
            }
        
        return self.user_preferences[user_id]
    
    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user alert preferences"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id].update(preferences)
        
        # Store to persistent storage
        self._save_user_preferences(user_id)
    
    def _save_user_preferences(self, user_id: str):
        """Save user preferences to storage"""
        try:
            if self.gcs:
                path = f"alert_preferences/{user_id}.json"
                self.gcs.upload_json(self.user_preferences[user_id], path)
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
    
    async def check_and_send_alerts(self, signals: List[TradingSignal], user_id: str):
        """Check signals and send alerts if criteria met"""
        
        preferences = self.get_user_preferences(user_id)
        
        # Check quiet hours
        if self._in_quiet_hours(preferences):
            logger.info(f"In quiet hours for user {user_id}, skipping alerts")
            return
        
        for signal in signals:
            # Check signal strength threshold
            min_strength = SignalStrength[preferences.get('min_strength', 'MODERATE')]
            
            if self._compare_signal_strength(signal.strength, min_strength) < 0:
                continue
            
            # Check specific tickers filter
            specific_tickers = preferences.get('specific_tickers')
            if specific_tickers and signal.ticker not in specific_tickers:
                continue
            
            # Send alert
            await self.send_signal_alert(signal, user_id)
    
    async def send_signal_alert(self, signal: TradingSignal, user_id: str):
        """Send alert for a specific signal"""
        
        preferences = self.get_user_preferences(user_id)
        
        # Format alert
        alert = self._format_alert(signal, preferences.get('format', 'detailed'))
        
        # Send through enabled channels
        for channel_name in preferences.get('channels', ['email']):
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                
                # Check if channel is enabled
                if preferences.get(f'{channel_name}_enabled', True):
                    success = await channel.send(alert, user_id)
                    
                    # Log alert
                    self._log_alert(signal, user_id, channel_name, success)
    
    async def send_test_alert(self, signal: TradingSignal, user_id: str, 
                             channel: str) -> bool:
        """Send test alert to specific channel"""
        
        if channel not in self.channels:
            logger.error(f"Unknown channel: {channel}")
            return False
        
        alert = self._format_alert(signal, 'detailed')
        alert['title'] = '🧪 TEST ALERT: ' + alert['title']
        
        return await self.channels[channel].send(alert, user_id)
    
    def _format_alert(self, signal: TradingSignal, format_type: str) -> Dict:
        """Format signal alert"""
        
        if format_type == 'minimal':
            return {
                'title': f"🚨 {signal.ticker}: {signal.strength.value} Signal",
                'message': signal.recommendation,
                'data': {
                    'signal_id': signal.signal_id,
                    'ticker': signal.ticker,
                    'confidence': signal.confidence
                }
            }
        
        else:  # detailed
            return {
                'title': f"📊 Trading Signal: {signal.ticker}",
                'message': f"""
Signal Type: {signal.signal_type.value}
Strength: {signal.strength.value} ({signal.confidence:.1%} confidence)
                
Entry: ${signal.targets.entry:.2f}
Stop Loss: ${signal.targets.stop_loss:.2f} (-{signal.targets.risk_percent:.1f}%)
Target 1: ${signal.targets.target_1:.2f} (+{((signal.targets.target_1/signal.targets.entry - 1) * 100):.1f}%)
Target 2: ${signal.targets.target_2:.2f} (+{((signal.targets.target_2/signal.targets.entry - 1) * 100):.1f}%)

Risk/Reward: {signal.risk_reward_ratio:.1f}:1
Time Horizon: {signal.time_horizon.value}
Expected Breakout: {signal.expected_breakout_days} days

{signal.recommendation}

Action Items:
{chr(10).join('• ' + item for item in signal.action_items)}

Watch For:
{chr(10).join('• ' + condition for condition in signal.watch_conditions)}
""",
                'data': signal.to_dict()
            }
    
    def _compare_signal_strength(self, strength1: SignalStrength, 
                                strength2: SignalStrength) -> int:
        """Compare signal strengths"""
        
        strength_order = [
            SignalStrength.NEUTRAL,
            SignalStrength.WEAK,
            SignalStrength.MODERATE,
            SignalStrength.STRONG,
            SignalStrength.VERY_STRONG
        ]
        
        idx1 = strength_order.index(strength1)
        idx2 = strength_order.index(strength2)
        
        return idx1 - idx2
    
    def _in_quiet_hours(self, preferences: Dict) -> bool:
        """Check if currently in quiet hours"""
        
        quiet_start = preferences.get('quiet_hours_start')
        quiet_end = preferences.get('quiet_hours_end')
        
        if quiet_start is None or quiet_end is None:
            return False
        
        current_hour = datetime.now().hour
        
        if quiet_start <= quiet_end:
            return quiet_start <= current_hour < quiet_end
        else:  # Spans midnight
            return current_hour >= quiet_start or current_hour < quiet_end
    
    def _log_alert(self, signal: TradingSignal, user_id: str, 
                  channel: str, success: bool):
        """Log alert for audit trail"""
        
        alert_log = {
            'timestamp': datetime.now().isoformat(),
            'signal_id': signal.signal_id,
            'ticker': signal.ticker,
            'user_id': user_id,
            'channel': channel,
            'success': success
        }
        
        self.alert_history.append(alert_log)
        
        # Limit history size
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-10000:]
        
        # Store to persistent storage
        if self.gcs and success:
            try:
                date_str = datetime.now().strftime('%Y%m%d')
                path = f"alert_logs/{date_str}/{signal.signal_id}_{channel}.json"
                self.gcs.upload_json(alert_log, path)
            except Exception as e:
                logger.error(f"Failed to log alert: {e}")
    
    async def send_daily_summary(self, user_id: str):
        """Send daily summary of signals"""
        
        # Get today's signals
        today_signals = [
            s for s in self.alert_history
            if s['user_id'] == user_id and
            datetime.fromisoformat(s['timestamp']).date() == datetime.now().date()
        ]
        
        if not today_signals:
            return
        
        # Create summary
        summary = {
            'title': f"📈 Daily Signal Summary - {datetime.now().strftime('%Y-%m-%d')}",
            'message': f"""
Today's Signal Activity:
- Total Signals: {len(today_signals)}
- Unique Tickers: {len(set(s['ticker'] for s in today_signals))}

Top Signals:
{self._format_top_signals_summary(today_signals)}

Visit your dashboard for full details.
            """,
            'data': {
                'date': datetime.now().isoformat(),
                'signal_count': len(today_signals)
            }
        }
        
        # Send through email
        await self.channels['email'].send(summary, user_id)
    
    def _format_top_signals_summary(self, signals: List[Dict]) -> str:
        """Format top signals for summary"""
        
        # Group by ticker
        by_ticker = {}
        for signal in signals:
            ticker = signal['ticker']
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(signal)
        
        # Format top 5
        summary_lines = []
        for ticker in list(by_ticker.keys())[:5]:
            count = len(by_ticker[ticker])
            summary_lines.append(f"• {ticker}: {count} signal{'s' if count > 1 else ''}")
        
        return '\n'.join(summary_lines)
