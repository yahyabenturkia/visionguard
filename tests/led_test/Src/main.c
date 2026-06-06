#include <stdint.h>

#define RCC_AHB1ENR   (*((volatile uint32_t*)0x40023830))
#define GPIOA_MODER   (*((volatile uint32_t*)0x40020000))
#define GPIOA_ODR     (*((volatile uint32_t*)0x40020014))

void delay_ms(uint32_t ms) {
    volatile uint32_t i;
    for(i = 0; i < ms * 3195; i++);
}

int main(void) {
    RCC_AHB1ENR |= (1U << 0);

    GPIOA_MODER &= ~(3U << 10);
    GPIOA_MODER |=  (1U << 10);

    while(1) {
        GPIOA_ODR |=  (1U << 5);
        delay_ms(1000);
        GPIOA_ODR &= ~(1U << 5);
        delay_ms(1000);
    }
}
