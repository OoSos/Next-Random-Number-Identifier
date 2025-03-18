export class NumberHandler {
  private lastNumber: number | null = null;

  public identifyNextNumber(newNumber: number): string {
    if (this.lastNumber === null) {
      this.lastNumber = newNumber;
      return `First number detected: ${newNumber}`;
    }

    const difference = newNumber - this.lastNumber;
    this.lastNumber = newNumber;

    if (difference > 0) {
      return `Higher than previous by ${difference}`;
    } else if (difference < 0) {
      return `Lower than previous by ${Math.abs(difference)}`;
    } else {
      return 'Same as previous number';
    }
  }

  public resetLastNumber(): void {
    this.lastNumber = null;
  }

  public getLastNumber(): number | null {
    return this.lastNumber;
  }
}
