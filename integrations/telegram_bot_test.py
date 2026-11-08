"""Telegram botni tekshirish (D-ilova): SOC_BOT_TOKEN va SOC_CHAT_ID muhit o‘zgaruvchilari bilan sinov xabari.

    set SOC_BOT_TOKEN=123:AA...   &  set SOC_CHAT_ID=-100123...   (Windows)
    python integrations/telegram_bot_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentinel.channels import Channels  # noqa: E402


class _S:
    def settings(self):
        return {}


if __name__ == "__main__":
    ch = Channels(_S(), "ai-alerts.json")
    st, err = ch.to_telegram("SentinelAI: sinov xabari. Bot ishlayapti.", ch.cfg())
    print(st, err)
