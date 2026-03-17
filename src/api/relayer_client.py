"""
Polymarket Relayer Client.
Uses the official py-builder-relayer-client package for merge/split/redeem operations.
Gas fees are handled automatically by the Polymarket relayer.
"""

import asyncio
import logging
from typing import Optional

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, SafeTransaction
from py_builder_signing_sdk.config import BuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
from web3 import Web3

from config.settings import settings
from models.data_models import Position, TokenType

logger = logging.getLogger(__name__)


# Polymarket contract addresses on Polygon
CONTRACTS = {
    "CTF": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
    "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
}

# Null parent collection ID (top-level conditions)
NULL_PARENT = bytes(32)

# Binary partition: [1, 2] = [YES outcome, NO outcome]
BINARY_PARTITION = [1, 2]

MAX_UINT256 = 2**256 - 1

# Minimal ABIs for contract interactions
CTF_ABI = [
    {
        "name": "splitPosition",
        "type": "function",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "partition", "type": "uint256[]"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "mergePositions",
        "type": "function",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "partition", "type": "uint256[]"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "redeemPositions",
        "type": "function",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    },
]

ERC20_APPROVE_ABI = [
    {
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"type": "bool"}],
    },
]


class RelayerClient:
    """
    Client for Polymarket Relayer API using py-builder-relayer-client.

    The Relayer enables:
    - Merging YES/NO positions to reclaim USDC
    - Splitting USDC into YES/NO positions
    - Position redemption after market resolution

    This is critical for capital recycling in the arbitrage strategy.
    """

    def __init__(self):
        builder_config = BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=settings.api.api_key,
                secret=settings.api.api_secret,
                passphrase=settings.api.api_passphrase,
            )
        )

        self._client = RelayClient(
            relayer_url=settings.api.relayer_host,
            chain_id=settings.chain_id,
            private_key=settings.wallet.private_key,
            builder_config=builder_config,
        )

        self._w3 = Web3()
        self._ctf_contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["CTF"]),
            abi=CTF_ABI,
        )
        self._usdc_contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACTS["USDC"]),
            abi=ERC20_APPROVE_ABI,
        )
        self._deployed = False

    async def connect(self):
        """Deploy the Safe wallet if not already deployed."""
        if not self._deployed:
            try:
                response = await asyncio.to_thread(self._client.deploy)
                await asyncio.to_thread(response.wait)
                logger.info("Safe wallet deployed")
            except Exception as e:
                logger.info(f"Safe deploy skipped (may already exist): {e}")
            self._deployed = True

    async def close(self):
        """No persistent connection to close."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def _to_condition_bytes(self, condition_id: str) -> bytes:
        """Convert a hex condition ID string to bytes32."""
        return bytes.fromhex(condition_id.removeprefix("0x").zfill(64))

    def _encode_merge(self, condition_id: str, amount: int) -> str:
        """Encode a mergePositions call."""
        return self._ctf_contract.encode_abi(
            abi_element_identifier="mergePositions",
            args=[
                Web3.to_checksum_address(CONTRACTS["USDC"]),
                NULL_PARENT,
                self._to_condition_bytes(condition_id),
                BINARY_PARTITION,
                amount,
            ],
        )

    def _encode_split(self, condition_id: str, amount: int) -> str:
        """Encode a splitPosition call."""
        return self._ctf_contract.encode_abi(
            abi_element_identifier="splitPosition",
            args=[
                Web3.to_checksum_address(CONTRACTS["USDC"]),
                NULL_PARENT,
                self._to_condition_bytes(condition_id),
                BINARY_PARTITION,
                amount,
            ],
        )

    def _encode_redeem(self, condition_id: str) -> str:
        """Encode a redeemPositions call."""
        return self._ctf_contract.encode_abi(
            abi_element_identifier="redeemPositions",
            args=[
                Web3.to_checksum_address(CONTRACTS["USDC"]),
                NULL_PARENT,
                self._to_condition_bytes(condition_id),
                BINARY_PARTITION,
            ],
        )

    def _encode_approve(self, spender: str, amount: int) -> str:
        """Encode an ERC20 approve call."""
        return self._usdc_contract.encode_abi(
            abi_element_identifier="approve",
            args=[Web3.to_checksum_address(spender), amount],
        )

    def _make_tx(self, to: str, data: str) -> SafeTransaction:
        """Create a SafeTransaction for a contract call."""
        return SafeTransaction(
            to=to,
            operation=OperationType.Call,
            data=data,
            value="0",
        )

    async def _execute(self, transactions: list[SafeTransaction], description: str) -> dict:
        """Execute transactions through the relayer."""
        try:
            response = await asyncio.to_thread(
                self._client.execute, transactions, description
            )
            tx_hash = response.transaction_hash

            # Wait for the transaction to be mined/confirmed
            await asyncio.to_thread(response.wait)

            logger.info(f"{description}: tx={tx_hash}")
            return {
                "success": True,
                "tx_hash": tx_hash,
            }

        except Exception as e:
            logger.error(f"{description} failed: {e}")
            return {"success": False, "error": str(e)}

    async def merge_positions(self, condition_id: str, amount: float) -> dict:
        """
        Merge equal YES and NO positions to reclaim USDC.

        1 YES token + 1 NO token → 1 USDC

        Args:
            condition_id: The market condition ID
            amount: Number of share pairs to merge

        Returns:
            Transaction result with tx_hash and USDC received
        """
        if amount <= 0:
            raise ValueError("Amount must be positive")

        amount_units = int(amount * 1_000_000)
        tx = self._make_tx(CONTRACTS["CTF"], self._encode_merge(condition_id, amount_units))

        result = await self._execute([tx], f"Merge {amount} shares for {condition_id}")
        if result["success"]:
            result["usdc_received"] = amount
            result["shares_merged"] = amount
        return result

    async def split_position(self, condition_id: str, amount: float) -> dict:
        """
        Split USDC into YES and NO positions.

        1 USDC → 1 YES token + 1 NO token

        Args:
            condition_id: The market condition ID
            amount: Amount of USDC to split

        Returns:
            Transaction result
        """
        if amount <= 0:
            raise ValueError("Amount must be positive")

        amount_units = int(amount * 1_000_000)
        tx = self._make_tx(CONTRACTS["CTF"], self._encode_split(condition_id, amount_units))

        result = await self._execute([tx], f"Split {amount} USDC for {condition_id}")
        if result["success"]:
            result["usdc_spent"] = amount
            result["yes_tokens"] = amount
            result["no_tokens"] = amount
        return result

    async def redeem_positions(self, condition_id: str) -> dict:
        """
        Redeem positions after market resolution.

        After a market resolves:
        - Winning tokens → 1 USDC each
        - Losing tokens → 0 USDC

        Args:
            condition_id: The resolved market condition ID

        Returns:
            Transaction result
        """
        tx = self._make_tx(CONTRACTS["CTF"], self._encode_redeem(condition_id))
        return await self._execute([tx], f"Redeem positions for {condition_id}")

    async def approve_ctf_spending(self, amount: Optional[float] = None) -> dict:
        """
        Approve the CTF contract to spend USDC (required before splitting).

        Args:
            amount: USDC amount to approve, or None for max approval.
        """
        approve_amount = int(amount * 1_000_000) if amount else MAX_UINT256
        tx = self._make_tx(CONTRACTS["USDC"], self._encode_approve(CONTRACTS["CTF"], approve_amount))
        return await self._execute([tx], "Approve USDC for CTF")

    def get_safe_address(self) -> str:
        """Get the expected Safe wallet address for this signer."""
        return self._client.get_expected_safe()


class CapitalRecycler:
    """
    Automated capital recycling through position merging.

    Monitors positions and automatically merges when:
    1. YES and NO positions are equal
    2. Combined cost basis < 1 (profitable)
    3. Amount exceeds merge threshold
    """

    def __init__(self, relayer: RelayerClient):
        self.relayer = relayer
        self.merge_threshold = settings.trading.auto_merge_threshold
        self._running = False

    async def check_and_merge(
        self,
        condition_id: str,
        yes_position: Position,
        no_position: Position,
    ) -> Optional[dict]:
        """
        Check if positions can be merged and execute if profitable.

        Returns merge result if executed, None otherwise.
        """
        mergeable = min(yes_position.size, no_position.size)

        if mergeable < self.merge_threshold:
            return None

        combined_cost = yes_position.avg_cost + no_position.avg_cost

        if combined_cost >= 1.0:
            logger.debug(f"Merge not profitable: cost basis {combined_cost:.4f}")
            return None

        profit = mergeable * (1.0 - combined_cost)

        logger.info(
            f"Merging {mergeable:.2f} shares at cost basis {combined_cost:.4f}, "
            f"expected profit: ${profit:.2f}"
        )

        result = await self.relayer.merge_positions(condition_id, mergeable)

        if result.get("success"):
            result["profit"] = profit
            result["cost_basis"] = combined_cost

        return result

    async def auto_recycle_loop(self, positions_callback, interval: float = 5.0):
        """
        Continuous loop checking for merge opportunities.

        Args:
            positions_callback: Async function that returns current positions
            interval: Check interval in seconds
        """
        self._running = True

        while self._running:
            try:
                positions = await positions_callback()

                market_positions = {}
                for pos in positions:
                    market_id = pos.market_id
                    if market_id not in market_positions:
                        market_positions[market_id] = {}
                    market_positions[market_id][pos.token_type] = pos

                for market_id, pos_dict in market_positions.items():
                    yes_pos = pos_dict.get(TokenType.YES)
                    no_pos = pos_dict.get(TokenType.NO)

                    if yes_pos and no_pos:
                        await self.check_and_merge(market_id, yes_pos, no_pos)

            except Exception as e:
                logger.error(f"Error in auto recycle loop: {e}")

            await asyncio.sleep(interval)

    def stop(self):
        """Stop the auto recycle loop."""
        self._running = False
