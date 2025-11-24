"""
Drug Information Service Module
Retrieves drug information from RxNorm and OpenFDA APIs
"""
import requests
from typing import Dict, List
from langchain.tools import tool

from ..config import DRUG_API_URLS
from ..schemas import DrugInfo, DrugInteraction


class DrugInfoService:
    """
    Retrieves drug information from external APIs
    """
    
    def __init__(self):
        """Initialize drug info service"""
        self.rxnorm_base = DRUG_API_URLS['RXNORM_BASE']
        self.openfda_base = DRUG_API_URLS['OPENFDA_BASE']
        self.timeout = DRUG_API_URLS['RXNORM_TIMEOUT']
    
    def get_rxcui(self, drug_name: str) -> str:
        """
        Get RxCUI (RxNorm Concept Unique Identifier) for a drug
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            RxCUI string or None
        """
        url = f"{self.rxnorm_base}/rxcui.json?name={drug_name}"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            rxcui_list = data.get('idGroup', {}).get('rxnormId', [])
            
            return rxcui_list[0] if rxcui_list else None
        
        except Exception as e:
            print(f"Error getting RxCUI for {drug_name}: {e}")
            return None
    
    def get_drug_purpose(self, rxcui: str) -> str:
        """
        Get therapeutic purpose/class of drug
        
        Args:
            rxcui: RxNorm Concept Unique Identifier
            
        Returns:
            Purpose/therapeutic class string
        """
        url = f"{self.rxnorm_base}/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                return "Not found"
            
            data = response.json()
            class_membership = data.get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', [])
            
            if class_membership:
                # Get first 3 classes for conciseness
                classes = [
                    info['rxclassMinConceptItem']['className'] 
                    for info in class_membership[:3]
                ]
                return ", ".join(classes)
            
            return "Not found"
        
        except Exception as e:
            print(f"Error getting drug purpose: {e}")
            return "Not found"
    
    def get_drug_side_effects(self, drug_name: str) -> str:
        """
        Get common side effects from OpenFDA
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Comma-separated side effects string
        """
        url = f"{self.openfda_base}/event.json?search=patient.drug.openfda.brand_name:{drug_name}&limit=3"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                return "Not found"
            
            data = response.json()
            results = data.get('results', [])
            
            side_effects = []
            for result in results:
                reactions = result.get('patient', {}).get('reaction', [])
                for r in reactions:
                    reaction_name = r.get('reactionmeddrapt')
                    if reaction_name:
                        side_effects.append(reaction_name)
            
            # Deduplicate and limit to top 5
            side_effects = list(set(side_effects))[:5]
            
            return ", ".join(side_effects) if side_effects else "Not found"
        
        except Exception as e:
            print(f"Error getting side effects: {e}")
            return "Not found"
    
    def get_drug_info(self, drug_name: str) -> DrugInfo:
        """
        Get comprehensive drug information
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            DrugInfo object with purpose, side effects, etc.
        """
        # Get RxCUI
        rxcui = self.get_rxcui(drug_name)
        
        if not rxcui:
            return DrugInfo(
                drug_name=drug_name,
                error=f"No RxNorm ID found for {drug_name}"
            )
        
        # Get purpose
        purpose = self.get_drug_purpose(rxcui)
        
        # Get side effects
        side_effects = self.get_drug_side_effects(drug_name)
        
        return DrugInfo(
            drug_name=drug_name,
            purpose=purpose,
            side_effects=side_effects
        )
    
    def get_drug_interactions(self, rxcui1: str, rxcui2: str) -> List[Dict]:
        """
        Get drug-drug interactions between two drugs
        
        Args:
            rxcui1: RxCUI of first drug
            rxcui2: RxCUI of second drug
            
        Returns:
            List of interaction dictionaries
        """
        url = f"{self.rxnorm_base}/interaction/interaction.json?rxcui={rxcui1}&rxcui={rxcui2}"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            interaction_list = data.get('interactionTypeGroup', [])
            
            interactions = []
            for group in interaction_list:
                for interaction_type in group.get('interactionType', []):
                    for interaction_pair in interaction_type.get('interactionPair', []):
                        description = interaction_pair.get('description', '')
                        severity = interaction_pair.get('severity', 'Unknown')
                        
                        if description:
                            interactions.append({
                                'description': description,
                                'severity': severity
                            })
            
            return interactions
        
        except Exception as e:
            print(f"Error getting interactions: {e}")
            return []
    
    def check_all_interactions(self, drug_list: List[str]) -> List[DrugInteraction]:
        """
        Check interactions between all drugs in a list
        
        Args:
            drug_list: List of drug names
            
        Returns:
            List of DrugInteraction objects
        """
        if len(drug_list) < 2:
            return []
        
        # Get RxCUIs for all drugs
        rxcui_map = {}
        for drug in drug_list:
            rxcui = self.get_rxcui(drug)
            if rxcui:
                rxcui_map[drug] = rxcui
        
        # Check interactions between each pair
        interactions = []
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i+1:]:
                if drug1 in rxcui_map and drug2 in rxcui_map:
                    drug_interactions = self.get_drug_interactions(
                        rxcui_map[drug1], 
                        rxcui_map[drug2]
                    )
                    
                    for interaction in drug_interactions:
                        interactions.append(DrugInteraction(
                            drug1=drug1,
                            drug2=drug2,
                            description=interaction['description'],
                            severity=interaction['severity']
                        ))
        
        return interactions


# LangChain tool wrappers
@tool
def get_drug_info_tool(drug_name: str) -> dict:
    """
    Gets drug purpose and side effects using RxNorm, RxClass, and OpenFDA.
    Returns a dictionary with purpose and side effects or error messages.
    """
    service = DrugInfoService()
    info = service.get_drug_info(drug_name)
    return info.dict()


@tool
def get_drug_interactions_tool(drug_list: List[str]) -> dict:
    """
    Gets drug-drug interactions using RxNorm.
    Returns a dictionary with interaction information.
    """
    service = DrugInfoService()
    interactions = service.check_all_interactions(drug_list)
    
    return {
        "interactions": [i.dict() for i in interactions],
        "error": None if interactions else "No interactions found or insufficient data"
    }


# Convenience functions
def get_drug_information(drug_name: str) -> DrugInfo:
    """Get drug information"""
    service = DrugInfoService()
    return service.get_drug_info(drug_name)


def check_drug_interactions(drug_list: List[str]) -> List[DrugInteraction]:
    """Check drug interactions"""
    service = DrugInfoService()
    return service.check_all_interactions(drug_list)